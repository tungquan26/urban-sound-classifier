const ProbabilityChart = {
    chartInstance: null,

    init() {
        const ctx = document.getElementById('probabilityChart').getContext('2d');

        Chart.defaults.font.family = "'Kalam', cursive";
        Chart.defaults.color = '#2d2d2d';

        this.chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['---', '---', '---', '---'],
                datasets: [{
                    label: 'Confidence (%)',
                    data: [0, 0, 0, 0],
                    backgroundColor: ['#ff4d4d', '#2d5da1', '#2d5da1', '#2d5da1'],
                    borderColor: '#2d2d2d',
                    borderWidth: 3,
                    borderRadius: 8,
                    borderSkipped: false,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(45,45,45,0.1)' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { font: { size: 16, weight: 'bold' } }
                    }
                },
                plugins: {
                    legend: { display: false }
                },
                animation: {
                    duration: 800,
                    easing: 'easeOutBounce'
                }
            }
        });
    },

    update(labels, probabilities) {
        this.chartInstance.data.labels = labels;
        this.chartInstance.data.datasets[0].data = probabilities;

        this.chartInstance.data.datasets[0].backgroundColor = probabilities.map((_, i) => i === 0 ? '#ff4d4d' : '#2d5da1');
        this.chartInstance.update();
    }
};
