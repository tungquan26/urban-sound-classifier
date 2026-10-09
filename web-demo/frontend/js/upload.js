const UploadManager = {
    file: null,

    init(onFileSelected) {
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const btnBrowse = document.getElementById('btn-browse');

        btnBrowse.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('click', (e) => {
            if (e.target !== btnBrowse) fileInput.click();
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('bg-muted');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('bg-muted'));

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('bg-muted');
            if (e.dataTransfer.files.length) {
                this.handleFile(e.dataTransfer.files[0], onFileSelected);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                this.handleFile(e.target.files[0], onFileSelected);
            }
        });
    },

    handleFile(file, callback) {

        if (!file.type.includes('audio/') && !file.name.match(/\.(wav|mp3)$/i)) {
            alert('Vui lòng tải lên file âm thanh (WAV, MP3)');
            return;
        }

        this.file = file;

        document.getElementById('file-name').textContent = file.name;
        document.getElementById('file-type').textContent = file.type || 'audio/wav';
        document.getElementById('file-size').textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
        document.getElementById('file-sample').textContent = '44.1 kHz';

        if (callback) callback(file);
    }
};
