const AudioVisualizer = {
    wavesurfer: null,

    init() {

        this.wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: '#2d2d2d',
            progressColor: '#ff4d4d',
            cursorColor: '#ff4d4d',
            barWidth: 3,
            barRadius: 3,
            cursorWidth: 2,
            height: 'auto',
            normalize: true,
        });

        const btnPlayPause = document.getElementById('btn-play-pause');
        const iconPlay = document.getElementById('icon-play');
        const iconPause = document.getElementById('icon-pause');
        const timeCurrent = document.getElementById('time-current');
        const timeTotal = document.getElementById('time-total');

        btnPlayPause.addEventListener('click', () => {
            if (this.wavesurfer.getDuration() > 0) {
                this.wavesurfer.playPause();
            }
        });

        this.wavesurfer.on('play', () => {
            iconPlay.classList.add('hidden');
            iconPause.classList.remove('hidden');
        });

        this.wavesurfer.on('pause', () => {
            iconPlay.classList.remove('hidden');
            iconPause.classList.add('hidden');
        });

        this.wavesurfer.on('audioprocess', () => {
            timeCurrent.textContent = this.formatTime(this.wavesurfer.getCurrentTime());
        });

        this.wavesurfer.on('ready', () => {
            const duration = this.wavesurfer.getDuration();
            timeTotal.textContent = this.formatTime(duration);
            document.getElementById('file-duration').textContent = this.formatTime(duration) + 's';
        });
    },

    load(file) {
        const objectUrl = URL.createObjectURL(file);
        this.wavesurfer.load(objectUrl);
    },

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
};
