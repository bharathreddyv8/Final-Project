/**
 * Evaluation Metrics Page - JavaScript
 * Author: Bharath Kumar
 */

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Chart instance
let metricsChart = null;

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    loadEvaluationMetrics();
});

/**
 * Load and display evaluation metrics
 */
async function loadEvaluationMetrics() {
    try {
        // Try to fetch from API first
        const response = await fetch(`${API_BASE_URL}/evaluation-metrics`);
        const data = await response.json();
        
        if (data.success && data.metrics) {
            displayMetrics(data.metrics);
            createMetricsChart(data.metrics);
            return;
        }
    } catch (error) {
        console.log('API not available, loading from local file:', error);
    }
    
    // Fallback: load from local file
    try {
        const response = await fetch('real_model_metrics.json');
        const allMetrics = await response.json();
        
        // Use Hybrid Model metrics
        const metrics = {
            accuracy: allMetrics["Hybrid Model"]["Accuracy"],
            precision: allMetrics["Hybrid Model"]["Precision"],
            recall: allMetrics["Hybrid Model"]["Recall"],
            f1_score: allMetrics["Hybrid Model"]["F1 Score"],
            roc_auc: allMetrics["Hybrid Model"]["ROC-AUC"]
        };
        
        displayMetrics(metrics);
        createMetricsChart(metrics);
    } catch (error) {
        console.error('Error loading evaluation metrics:', error);
        showError('Failed to load evaluation metrics. Make sure the backend server is running or check the local metrics file.');
    }
}

/**
 * Display metrics values and progress bars
 */
function displayMetrics(metrics) {
    // Format percentage
    const formatPercent = (value) => (value * 100).toFixed(2) + '%';
    
    // Accuracy
    document.getElementById('accuracyValue').textContent = formatPercent(metrics.accuracy);
    animateProgressBar('accuracyBar', metrics.accuracy * 100);
    
    // Precision
    document.getElementById('precisionValue').textContent = formatPercent(metrics.precision);
    animateProgressBar('precisionBar', metrics.precision * 100);
    
    // Recall
    document.getElementById('recallValue').textContent = formatPercent(metrics.recall);
    animateProgressBar('recallBar', metrics.recall * 100);
    
    // F1-Score
    document.getElementById('f1Value').textContent = formatPercent(metrics.f1_score);
    animateProgressBar('f1Bar', metrics.f1_score * 100);
    
    // ROC-AUC
    document.getElementById('aucValue').textContent = formatPercent(metrics.roc_auc);
    animateProgressBar('aucBar', metrics.roc_auc * 100);
}

/**
 * Animate progress bar
 */
function animateProgressBar(elementId, percentage) {
    const element = document.getElementById(elementId);
    if (element) {
        // Start at 0
        element.style.width = '0%';
        
        // Animate to target value
        setTimeout(() => {
            element.style.width = percentage + '%';
        }, 100);
    }
}

/**
 * Create metrics comparison chart
 */
function createMetricsChart(metrics) {
    const ctx = document.getElementById('metricsChart');
    
    if (!ctx) return;
    
    // Destroy existing chart if any
    if (metricsChart) {
        metricsChart.destroy();
    }
    
    // Prepare data
    const labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'];
    const values = [
        metrics.accuracy * 100,
        metrics.precision * 100,
        metrics.recall * 100,
        metrics.f1_score * 100,
        metrics.roc_auc * 100
    ];
    
    // Create chart
    metricsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Performance (%)',
                data: values,
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(139, 92, 246, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(236, 72, 153, 0.8)'
                ],
                borderColor: [
                    'rgb(16, 185, 129)',
                    'rgb(59, 130, 246)',
                    'rgb(139, 92, 246)',
                    'rgb(245, 158, 11)',
                    'rgb(236, 72, 153)'
                ],
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.5,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    ticks: {
                        font: {
                            size: 12,
                            weight: '600'
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart'
            }
        }
    });
}

/**
 * Show error message
 */
function showError(message) {
    const metricsContainer = document.querySelector('.metrics-dashboard');
    if (metricsContainer) {
        metricsContainer.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 3rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                <h3 style="color: var(--danger-color); margin-bottom: 1rem;">${message}</h3>
                <p style="color: var(--text-secondary);">
                    Please run the training script first:
                    <code style="background: var(--surface-dark); padding: 0.5rem 1rem; border-radius: 0.375rem; display: inline-block; margin-top: 1rem;">
                        python backend/model/train.py
                    </code>
                </p>
            </div>
        `;
    }
}

// Add smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
