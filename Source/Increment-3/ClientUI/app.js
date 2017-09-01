(function localFileVideoPlayerInit(win) {
    var URL = win.URL || win.webkitURL,

        playSelectedFile = function playSelectedFileInit(event) {
            var file = this.files[0];

            var videoNode = document.querySelector('video');

            var fileURL = URL.createObjectURL(file);

            videoNode.src = fileURL;
        },
        inputNode = document.querySelector('input');

    inputNode.addEventListener('change', playSelectedFile, false);
}(window));