document.addEventListener('DOMContentLoaded', () => {

    AudioVisualizer.init();
    ProbabilityChart.init();

    UploadManager.init((file) => {

        AudioVisualizer.load(file);
    });

    const btnClassify = document.getElementById('btn-classify');

    btnClassify.addEventListener('click', async (e) => {

        e.preventDefault();

        if (!UploadManager.file) {
            alert('Vui lòng chọn một file âm thanh trước!');
            return;
        }

        const originalBtnHTML = btnClassify.innerHTML;
        btnClassify.innerHTML = `
            ANALYZING...
            <svg class="animate-spin h-6 w-6 ml-2" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
        `;
        btnClassify.disabled = true;

        const resultsSection = document.getElementById('analysis-results');
        if (resultsSection) {
            resultsSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }

        const result = await ApiService.classifyAudio(UploadManager.file);

        if (result) {

            document.getElementById('result-label').textContent = result.top_class.toUpperCase();
            document.getElementById('result-confidence').textContent = `${result.confidence}%`;

            const labels = result.distribution.map(d => d.class);
            const probs = result.distribution.map(d => d.prob);

            ProbabilityChart.update(labels, probs);
        }

        btnClassify.innerHTML = originalBtnHTML;
        btnClassify.disabled = false;
    });
});
