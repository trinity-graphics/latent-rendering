
import drjit as dr
import mitsuba as mi


class LatentIntegrator(mi.ad.integrators.common.RBIntegrator):
    """
    A PRB-style integrator allowing for latent rendering.

    Behaviour is selected at construction time via two flags:

    - ``ambient_term``    : compute the BSDF ``ambient`` (+ ``occ_ambient``) term.
    - ``separate_terms``  : when True, the ambient term is written to its own
                            AOV (gated to the first non-delta vertex); when
                            False, it is folded directly into the reflected
                            radiance ``Lr``.

    The flat background (formerly the "miss emitter") is driven by the
    differentiable ``miss_color`` parameter. The integrator only accumulates the
    (detached) throughput reaching gated miss events and emits it as the final
    ``missw`` AOV; the actual ``miss_weight * miss_color`` background is applied
    by :meth:`renderers.latent.LatentRenderer.render` at image resolution. This
    keeps ``miss_color`` differentiable without the cost of a post-loop,
    sample-resolution AD reduction (and without the symbolic loop severing its
    gradient). The miss emitter is no longer sampled.

    Flags may also be supplied through the scene description (``props``), which
    takes precedence over the constructor arguments.
    """

    def __init__(self,
                 props,
                 ambient_term: bool = False,
                 separate_terms: bool = False):
        super().__init__(props)

        self.ambient_term = bool(props.get("ambient_term", ambient_term))
        self.separate_terms = bool(props.get("separate_terms", separate_terms))

        # Differentiable flat background applied to rays that miss geometry.
        self.miss_color = mi.Spectrum(props.get("miss_color", 0.0))

    # Whether the ambient term is emitted as its own AOV (vs folded into Lr).
    @property
    def _ambient_aov(self) -> bool:
        return self.separate_terms and self.ambient_term

    def traverse(self, callback):
        callback.put("miss_color", self.miss_color, mi.ParamFlags.Differentiable)

    @dr.syntax
    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: mi.Spectrum | None,
               δaovs: mi.Spectrum | None = None,
               state_in: mi.Spectrum | None = None,
               active: mi.Bool = mi.Bool(True),
               **kwargs  # Absorbs unused arguments
    ) -> tuple[mi.Spectrum, mi.Bool, list[mi.Float], mi.Spectrum]:
        """
        See ``ADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        num_ch = len(mi.Spectrum())

        # Config: resolved to a Python boolean, constant-folded at trace time
        ambient_aov = self._ambient_aov

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if primal else state_in["L"])    # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance

        # Separate-AOV accumulator (+ its adjoint). Always allocated so
        # dr.syntax treats it as loop-carried state; it is only written /
        # returned when the ambient term is emitted as its own AOV.
        ambient = mi.Spectrum(0 if primal else state_in.get("ambient", 0))
        δambient = mi.Spectrum(0)
        if dr.hint(not primal and δaovs is not None and ambient_aov, mode='scalar'):
            δambient = mi.Spectrum(δaovs[0:num_ch])

        β = mi.Spectrum(1)                            # Path throughput weight

        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes

        miss_active = mi.Bool(True)                 # Active lanes for miss rays
        miss_weight = mi.Spectrum(0)                # Throughput reaching gated misses
        first_active = mi.Bool(True)

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        while dr.hint(active,
                      max_iterations=self.max_depth,
                      label="Path Replay Backpropagation (%s)" % mode.name):
            active_next = mi.Bool(active)

            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            with dr.resume_grad(when=not primal):
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All,
                                         coherent=(depth == 0))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Hide the environment emitter if necessary
            if dr.hint(self.hide_emitters, mode='scalar'):
                active_next &= ~((depth == 0) & ~si.is_valid())

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            mis = mi.ad.integrators.common.mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )

            # Flat background applied directly to rays that miss geometry.
            # Gated like the old miss emitter: only along delta/specular paths
            # from the camera (disabled once a non-delta surface is hit).
            do_miss = ~si.is_valid() & miss_active

            # Accumulate the (detached) throughput reaching each gated miss.
            # The flat background is ``miss_weight * miss_color``, applied at
            # image resolution by the renderer. Doing the ``* miss_color``
            # multiply here, inside the symbolic loop, is both impossible to
            # differentiate (the loop boundary severs the captured ``miss_color``
            # AD edge) and -- if applied via a post-loop ``backward_from`` --
            # expensive (a second sample-resolution AD reduction). Emitting the
            # weight as an AOV defers that reduction to the film.
            miss_weight += dr.select(do_miss, β, mi.Spectrum(0))

            with dr.resume_grad(when=not primal):
                # If a finite light is hit directly by the ray
                Le = β * mis * ds.emitter.eval(si, active_next)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            spec_surf = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            miss_active &= ~spec_surf

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= (ds.pdf != 0.0)

            em_val = scene.eval_emitter_direction(si, ds,  active_em)

            with dr.resume_grad(when=not primal):
                if dr.hint(not primal, mode='scalar'):
                    # Given the detached emitter sample, *recompute* its
                    # contribution with AD to enable light source optimization
                    ds.d = dr.replace_grad(ds.d, dr.normalize(ds.p - si.p))
                    em_val = scene.eval_emitter_direction(si, ds, active_em)
                    em_weight = dr.replace_grad(em_weight, dr.select((ds.pdf != 0), em_val / ds.pdf, 0))
                    dr.disable_grad(ds.d)

                # Evaluate BSDF * cos(theta) differentiably
                wo = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
                mis_em = dr.select(ds.delta, 1, mi.ad.integrators.common.mis_weight(ds.pdf, bsdf_pdf_em))

                # IF and Only IF we are doing light sampling this is the incoming radiance
                Lr_dir = β * mis_em * bsdf_value_em * em_weight

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)

            # ------------------ Secondary Lighting Term ------------------

            # IF SURFACE IS DIFFUSED or Used as diffused
            # we are doing light sampling, and when we do that
            # we can either fail or not.  This works as a single colour
            # adder for diffused surfaces, accounting for shadows via the
            # passed/failed light-sampling masks.
            #
            # When terms are tracked separately, gate to the first non-delta
            # vertex; when folded into Lr, accumulate along the path (matching
            # the historical behaviour of the non-AOV integrators).
            gate = first_active if self.separate_terms else mi.Bool(True)
            sec_mask = active_next & spec_surf & gate
            ocl_mask = ~dr.select(spec_surf, active_em, True) & gate
            first_active &= ~spec_surf

            with dr.resume_grad(when=not primal):
                # Ambient term (always computed; eval_attribute returns 0 when
                # the corresponding BSDF attribute is absent).
                if dr.hint(self.ambient_term, mode='scalar'):
                    sec_refl = bsdf.eval_attribute("ambient", si, sec_mask)
                    ocl_refl = bsdf.eval_attribute("occ_ambient", si, ocl_mask)
                    amb_hit = β * sec_refl + β * ocl_refl
                else:
                    amb_hit = dr.zeros_like(β)

                # Route ambient: own AOV or folded into Lr
                if dr.hint(not ambient_aov, mode='scalar'):
                    Lr_dir += amb_hit

            # ---- Update loop variables based on current interaction -----

            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)
            if dr.hint(ambient_aov, mode='scalar'):
                ambient = (ambient + amb_hit) if primal else (ambient - amb_hit)
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β *= (bsdf_weight)

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(dr.abs(β))
            active_next &= (β_max != 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            # β[rr_active] *= dr.rcp(rr_prob)
            β[rr_active & active_next] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            # ------------------ Differential phase only ------------------

            if dr.hint(not primal, mode='scalar'):
                with dr.resume_grad():
                    # 'L' stores the indirectly reflected radiance at the
                    # current vertex but does not track parameter derivatives.
                    # The following addresses this by canceling the detached
                    # BSDF value and replacing it with an equivalent term that
                    # has derivative tracking enabled. (nit picking: the
                    # direct/indirect terminology isn't 100% accurate here,
                    # since there may be a direct component that is weighted
                    # via multiple importance sampling)

                    # Recompute 'wo' to propagate derivatives to cosine term
                    wo = si.to_local(ray.d)

                    # Re-evaluate BSDF * cos(theta) differentiably
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next)

                    # Detached version of the above term and inverse
                    bsdf_val_det = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_det = dr.select(bsdf_val_det != 0,
                                                 dr.rcp(bsdf_val_det), 0)

                    # Differentiable version of the reflected indirect
                    # radiance. Minor optional tweak: indicate that the primal
                    # value of the second term is always 1.
                    tmp = inv_bsdf_val_det * bsdf_val
                    tmp_replaced = dr.replace_grad(dr.ones(mi.Float, dr.width(tmp)), tmp) #FIXME
                    Lr_ind = L * tmp_replaced

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + Lr_dir + Lr_ind

                    attached_contrib = dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo)
                    if dr.hint(attached_contrib, mode='scalar'):
                        raise Exception(
                            "The contribution computed by the differential "
                            "rendering phase is not attached to the AD graph! "
                            "Raising an exception since this is usually "
                            "indicative of a bug (for example, you may have "
                            "forgotten to call dr.enable_grad(..) on one of "
                            "the scene parameters, or you may be trying to "
                            "optimize a parameter that does not generate "
                            "derivatives in detached PRB.)")

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    if dr.hint(mode == dr.ADMode.Backward, mode='scalar'):
                        dr.backward_from(δL * Lo)
                        if dr.hint(ambient_aov, mode='scalar'):
                            dr.backward_from(δambient * (ambient + amb_hit))
                    else:
                        δL += dr.forward_to(Lo)
                        if dr.hint(ambient_aov, mode='scalar'):
                            δambient += dr.forward_to(ambient + amb_hit)

            depth[si.is_valid()] += 1
            active = active_next

        # ---------------- Assemble AOV outputs + state -------------------
        aovs = []
        state_out = {"L": L}
        if ambient_aov:
            src = ambient if primal else δambient
            aovs += [mi.Float(src[n]) for n in range(num_ch)]
            state_out["ambient"] = ambient

        # Flat-background weight: the detached throughput reaching gated misses,
        # always emitted as the final AOV. The renderer forms the background as
        # ``missw * miss_color`` (differentiating ``miss_color`` there).
        aovs += [mi.Float(dr.detach(miss_weight[n])) for n in range(num_ch)]

        return (
            L if primal else δL, # Radiance/differential radiance
            (depth != 0),        # Ray validity flag for alpha blending
            aovs,                # Latent AOVs (ambient, then missw)
            state_out            # State for the differential phase
        )

    def aov_names(self):
        names = []
        if self._ambient_aov:
            names += [f"ambient{n:02}" for n in range(len(mi.Spectrum()))]
        names += [f"missw{n:02}" for n in range(len(mi.Spectrum()))]
        return names

    def to_string(self):
        return (f"LatentIntegrator[max_depth={self.max_depth}, "
                f"rr_depth={self.rr_depth}, "
                f"ambient_term={self.ambient_term}, "
                f"separate_terms={self.separate_terms}]")
