var EXPLORER_FRAMES = 60;
var explorerFrames = [];
var trainingFrame = 0;

// The un-edited frame each scene/edit was trained on.
var TRAINING_FRAMES = {
  cbox:   {move_camera: 30, move_object: 0,  move_light: 30},
  lamp:   {move_camera: 0,  move_object: 0,  move_light: 0},
  veach:  {move_camera: 0,  move_object: 22, move_light: 0},
  dining: {move_camera: 0,  move_object: 30, move_light: 0},
  living: {move_camera: 24, move_object: 0,  move_light: 0},
};

// Preload all frames of the selected scene/edit so scrubbing is instant.
function loadExplorerCase() {
  var scene = $('#explorer-scene').val();
  var edit = $('#explorer-case').val();
  var dir = './static/explorer/' + scene + '/' + edit + '/';
  explorerFrames = [];
  for (var i = 0; i < EXPLORER_FRAMES; i++) {
    explorerFrames[i] = new Image();
    explorerFrames[i].src = dir + String(i).padStart(2, '0') + '.webp';
  }
  trainingFrame = TRAINING_FRAMES[scene][edit];
  $('#explorer-slider').val(trainingFrame);
  setExplorerFrame(trainingFrame);
}

function setExplorerFrame(i) {
  $('#explorer-image').attr('src', explorerFrames[i].src);
  // Always two lines, so the layout doesn't jump on the training frame.
  $('#explorer-frame').html('Frame ' + i + ' / ' + (EXPLORER_FRAMES - 1) + '<br>' +
                            (Number(i) === trainingFrame ? '(training view)' : '&nbsp;'));
}

$(document).ready(function() {
  $('#explorer-scene, #explorer-case').on('change', loadExplorerCase);
  $('#explorer-slider').on('input', function() { setExplorerFrame(this.value); });
  loadExplorerCase();
});
