/**
 * Medical Insurance Fraud Detection - Frontend JavaScript
 * Author: Bharath Kumar
 * Real-time API integration with visualization
 */

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global chart instance
let explanationChart = null;
let latestRecentPredictions = [];
let adminDashboardRefreshInterval = null;

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeForm();
    applyQueryParamsFromUrl();
    checkAPIHealth();
});

/**
 * Prefill form fields from URL query parameters if present.
 */
function applyQueryParamsFromUrl() {
    try {
        const params = new URLSearchParams(window.location.search);
        if (!params || [...params.keys()].length === 0) return;

        const form = document.getElementById('claimForm');
        if (!form) return;

        for (const [key, value] of params.entries()) {
            // Some params might be empty string by design
            const el = document.getElementById(key);
            if (!el) continue;

            // For numeric inputs and selects, set value directly
            if (el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA') {
                el.value = value;
            }
        }
    } catch (err) {
        console.warn('Failed to apply URL params to form:', err);
    }
}

/**
 * Initialize form event handlers
 */
function initializeForm() {
    const form = document.getElementById('claimForm');
    if (!form) return;
    form.addEventListener('submit', handleFormSubmit);
}

/**
 * Check API health status
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (!data.models_loaded) {
            showNotification('Warning: Models not loaded. Please train models first.', 'warning');
        }
    } catch (error) {
        showNotification('Error: Cannot connect to API. Make sure the backend is running.', 'error');
    }
}

/**
 * Handle form submission
 */
async function handleFormSubmit(event) {
    event.preventDefault();
    
    // Show loading state
    setLoadingState(true);
    hideResults();
    
    // Collect form data
    const claimData = getFormData();
    
    try {
        // Make prediction request
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(claimData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
        console.error('Prediction error:', error);
    } finally {
        setLoadingState(false);
    }
}

/**
 * Get form data as JSON object
 */
function getFormData() {
    const form = document.getElementById('claimForm');
    const formData = new FormData(form);
    
    const data = {};
    for (const [key, value] of formData.entries()) {
        // Convert to appropriate type
        if (key === 'claim_amount') {
            data[key] = parseFloat(value);
        } else if (key === 'patient_email') {
            // Only include patient_email when non-empty to avoid backend validation errors
            if (value && value.trim()) {
                data[key] = value.trim();
            }
        } else {
            data[key] = parseInt(value);
        }
    }
    
    return data;
}

/**
 * Display prediction results
 */
function displayResults(result) {
    // Show result cards
    document.getElementById('resultCard').style.display = 'block';
    document.getElementById('explanationCard').style.display = 'block';
    
    // Display main prediction
    displayPrediction(result);
    
    // Display risk meter
    displayRiskMeter(result);
    
    // Display model outputs
    displayModelOutputs(result);
    
    // Display explanation
    displayExplanation(result);
    
    // Scroll to results
    document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Keep admin dashboard in sync when open
    if (isAdminDashboardOpen()) {
        refreshAdminDashboardData();
    }

    // Notify user if backend sent an approval email
    if (result.email_sent) {
        const to = result.email_to || '';
        showNotification(`Approval email sent to ${to}`, 'success');
    } else if (result.email_error) {
        showNotification(`Email failed: ${result.email_error}`, 'warning');
    }
}

/**
 * Display main prediction badge
 */
function displayPrediction(result) {
    const badge = document.getElementById('resultBadge');
    const predictionText = document.getElementById('predictionResult');
    const confidenceText = document.getElementById('confidenceText');
    
    // Set prediction
    predictionText.textContent = result.final_prediction;
    
    // Set confidence
    const confidencePercent = (result.confidence * 100).toFixed(1);
    confidenceText.textContent = `${confidencePercent}% Confidence`;
    
    // Set badge style
    badge.className = 'result-badge';
    if (result.final_prediction === 'FRAUD') {
        badge.classList.add('fraud');
    } else {
        badge.classList.add('genuine');
    }
}

/**
 * Display risk meter
 */
function displayRiskMeter(result) {
    const riskValue = document.getElementById('riskValue');
    const meterFill = document.getElementById('meterFill');
    
    const riskPercent = (result.hybrid_risk_score * 100).toFixed(1);
    riskValue.textContent = `${riskPercent}%`;
    
    // Animate meter fill
    setTimeout(() => {
        meterFill.style.width = `${riskPercent}%`;
    }, 100);
}

/**
 * Display individual model outputs
 */
function displayModelOutputs(result) {
    // XGBoost
    const xgboostValue = document.getElementById('xgboostValue');
    const xgboostBar = document.getElementById('xgboostBar');
    const xgboostInterpretation = document.getElementById('xgboostInterpretation');
    const xgboostPercent = (result.xgboost_probability * 100).toFixed(1);
    
    xgboostValue.textContent = `${xgboostPercent}%`;
    setTimeout(() => {
        xgboostBar.style.width = `${xgboostPercent}%`;
    }, 200);
    
    // Add interpretation
    if (result.xgboost_probability < 0.3) {
        xgboostInterpretation.textContent = 'Low fraud probability - patterns match legitimate claims.';
        xgboostInterpretation.style.color = 'var(--success-color)';
    } else if (result.xgboost_probability < 0.7) {
        xgboostInterpretation.textContent = 'Moderate risk - some suspicious patterns detected.';
        xgboostInterpretation.style.color = 'var(--warning-color)';
    } else {
        xgboostInterpretation.textContent = 'High fraud probability - strong indicators of fraud.';
        xgboostInterpretation.style.color = 'var(--danger-color)';
    }
    
    // Isolation Forest
    const isoforestValue = document.getElementById('isoforestValue');
    const isoforestBar = document.getElementById('isoforestBar');
    const isoforestInterpretation = document.getElementById('isoforestInterpretation');
    const isoforestPercent = (result.isolation_forest_score * 100).toFixed(1);
    
    isoforestValue.textContent = `${isoforestPercent}%`;
    setTimeout(() => {
        isoforestBar.style.width = `${isoforestPercent}%`;
    }, 300);
    
    // Add interpretation
    if (result.isolation_forest_score < 0.3) {
        isoforestInterpretation.textContent = 'Normal claim - similar to typical patterns.';
        isoforestInterpretation.style.color = 'var(--success-color)';
    } else if (result.isolation_forest_score < 0.7) {
        isoforestInterpretation.textContent = 'Somewhat unusual - has atypical characteristics.';
        isoforestInterpretation.style.color = 'var(--warning-color)';
    } else {
        isoforestInterpretation.textContent = 'Highly anomalous - very different from normal claims.';
        isoforestInterpretation.style.color = 'var(--danger-color)';
    }
}

/**
 * Display explanation with SHAP values
 */
function displayExplanation(result) {
    // Display feature chart
    displayFeatureChart(result.explanation);
    
    // Display feature details
    displayFeatureDetails(result.explanation);
    
    // Display summary
    document.getElementById('explanationSummary').textContent = result.summary;
}

/**
 * Display feature contribution chart using Chart.js
 */
function displayFeatureChart(explanations) {
    const canvas = document.getElementById('explanationChart');
    const ctx = canvas.getContext('2d');
    
    // Destroy previous chart if exists
    if (explanationChart) {
        explanationChart.destroy();
    }
    
    // Prepare data
    const labels = explanations.map(exp => exp.feature);
    const data = explanations.map(exp => Math.abs(exp.impact));
    const colors = explanations.map(exp => 
        exp.impact > 0 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(16, 185, 129, 0.8)'
    );
    
    // Create chart
    explanationChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Feature Impact (Absolute SHAP Value)',
                data: data,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.8', '1')),
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Impact: ${context.parsed.x.toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    title: {
                        display: true,
                        text: 'Absolute Impact on Fraud Prediction'
                    }
                },
                y: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

/**
 * Display feature details list
 */
function displayFeatureDetails(explanations) {
    const container = document.getElementById('featureDetails');
    container.innerHTML = '';
    
    explanations.forEach(exp => {
        const item = document.createElement('div');
        item.className = 'feature-item';
        
        const impactClass = exp.impact > 0 ? 'impact-positive' : 'impact-negative';
        const impactText = exp.impact > 0 ? 'Increases Fraud Risk' : 'Decreases Fraud Risk';
        const arrow = exp.impact > 0 ? '↑' : '↓';
        
        item.innerHTML = `
            <span class="feature-name">${exp.feature}</span>
            <div class="feature-impact">
                <span class="impact-value ${impactClass}">
                    ${arrow} ${Math.abs(exp.impact).toFixed(4)}
                </span>
                <span style="font-size: 0.75rem; color: var(--text-muted);">
                    ${impactText}
                </span>
            </div>
        `;
        
        container.appendChild(item);
    });
}

/**
 * Show admin dashboard modal
 */
async function showAdminDashboard() {
    const modal = document.getElementById('adminModal');
    if (!modal) {
        console.warn('Admin modal element not found');
        return;
    }
    modal.classList.add('active');

    await refreshAdminDashboardData();

    if (adminDashboardRefreshInterval) {
        clearInterval(adminDashboardRefreshInterval);
    }
    adminDashboardRefreshInterval = setInterval(() => {
        if (!isAdminDashboardOpen()) return;
        refreshAdminDashboardData();
    }, 15000);
}

/**
 * Close admin dashboard modal
 */
function closeAdminDashboard() {
    const modal = document.getElementById('adminModal');
    if (!modal) return;
    modal.classList.remove('active');

    if (adminDashboardRefreshInterval) {
        clearInterval(adminDashboardRefreshInterval);
        adminDashboardRefreshInterval = null;
    }
}

async function refreshAdminDashboardData() {
    await loadStatistics();
    await loadRecentPredictions();
}

function isAdminDashboardOpen() {
    const modal = document.getElementById('adminModal');
    return !!(modal && modal.classList.contains('active'));
}

/**
 * Load prediction statistics
 */
async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE_URL}/statistics`);
        const data = await response.json();
        
        if (data.success && data.total_predictions > 0) {
            document.getElementById('statTotal').textContent = data.total_predictions;
            document.getElementById('statFraud').textContent = data.fraud_count;
            document.getElementById('statGenuine').textContent = data.genuine_count;
            document.getElementById('statFraudRate').textContent = `${data.fraud_rate.toFixed(1)}%`;
        } else {
            document.getElementById('statTotal').textContent = '0';
            document.getElementById('statFraud').textContent = '0';
            document.getElementById('statGenuine').textContent = '0';
            document.getElementById('statFraudRate').textContent = '0%';
        }
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

/**
 * Load recent predictions
 */
async function loadRecentPredictions() {
    try {
        const response = await fetch(`${API_BASE_URL}/recent-predictions?limit=20`);
        const data = await response.json();
        latestRecentPredictions = Array.isArray(data.predictions) ? data.predictions : [];
        
            const tbody = document.getElementById('predictionsTableBody');
        if (!tbody) return;
        tbody.innerHTML = '';
        if (latestRecentPredictions.length > 0) {
            latestRecentPredictions.forEach(pred => {
                const row = document.createElement('tr');
                const timestamp = new Date(pred.timestamp).toLocaleString();
                const dateStr = pred.timestamp.split('T')[0];
                const badgeClass = pred.prediction === 'FRAUD' ? 'fraud' : 'genuine';
                row.setAttribute('data-date', dateStr);
                row.setAttribute('data-prediction', pred.prediction);
                row.setAttribute('data-provider-id', pred.provider_id || '');
                row.setAttribute('data-patient-id', pred.patient_id || '');
                row.innerHTML = `
                    <td><strong>${pred.claim_id}</strong></td>
                    <td>${timestamp}</td>
                    <td>$${pred.claim_amount.toLocaleString('en-US', { minimumFractionDigits: 2 })}</td>
                    <td>${(pred.risk_score * 100).toFixed(1)}%</td>
                    <td><span class="badge ${badgeClass}">${pred.prediction}</span></td>
                `;
                tbody.appendChild(row);
            });
        } else {
            tbody.innerHTML = '<tr><td colspan="5" class="no-data">No predictions yet</td></tr>';
        }
    } catch (error) {
        console.error('Error loading recent predictions:', error);
        latestRecentPredictions = [];
    }
}

/**
 * Filter predictions by date range and prediction type
 */
function applyFilters() {
    const dateFrom = document.getElementById('filterDateFrom')?.value;
    const dateTo = document.getElementById('filterDateTo')?.value;
    const prediction = document.getElementById('filterPrediction')?.value || '';

    const tbody = document.getElementById('predictionsTableBody');
    if (!tbody) return;
    const rows = tbody.querySelectorAll('tr:not(.no-data-filtered)');
    let visibleCount = 0;
    rows.forEach(row => {
        const rowDateStr = row.dataset.date;
        const rowPrediction = row.dataset.prediction || '';
        let showRow = true;
        if (dateFrom && rowDateStr < dateFrom) showRow = false;
        if (dateTo && rowDateStr > dateTo) showRow = false;
        if (prediction && rowPrediction !== prediction) showRow = false;
        row.style.display = showRow ? '' : 'none';
        if (showRow) visibleCount++;
    });
    if (visibleCount === 0) {
        const noDataRow = tbody.querySelector('.no-data-filtered');
        if (!noDataRow) {
            const row = document.createElement('tr');
            row.className = 'no-data-filtered';
            row.innerHTML = '<td colspan="5" class="no-data">No predictions match the selected filters</td>';
            tbody.appendChild(row);
        }
    } else {
        const noDataRow = tbody.querySelector('.no-data-filtered');
        if (noDataRow) noDataRow.remove();
    }
    showNotification('Filters applied', 'info');
}

function clearFilters() {
    const filterDateFrom = document.getElementById('filterDateFrom');
    const filterDateTo = document.getElementById('filterDateTo');
    const filterPrediction = document.getElementById('filterPrediction');
    
    if (filterDateFrom) filterDateFrom.value = '';
    if (filterDateTo) filterDateTo.value = '';
    if (filterPrediction) filterPrediction.value = '';

    const tbody = document.getElementById('predictionsTableBody');
    if (!tbody) return;
    tbody.querySelectorAll('tr').forEach(row => {
        row.style.display = '';
    });
    const noDataRow = tbody.querySelector('.no-data-filtered');
    if (noDataRow) noDataRow.remove();

    showNotification('Filters cleared', 'info');
}


/**
 * Clear recent prediction history by calling backend DELETE endpoint
 */
async function clearHistory() {
    try {
        const resp = await fetch(`${API_BASE_URL}/recent-predictions`, {
            method: 'DELETE'
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to clear history');
        }

        const data = await resp.json();
        showNotification(data.message || 'History cleared', 'success');

        // Refresh admin dashboard data
        if (isAdminDashboardOpen()) {
            await refreshAdminDashboardData();
        }
    } catch (error) {
        console.error('Error clearing history:', error);
        showNotification(`Error: ${error.message}`, 'error');
    }
}

function filterByEntity(entityType, entityId) {
    const tbody = document.getElementById('predictionsTableBody');
    if (!tbody) return;

    const dataAttr = entityType === 'provider' ? 'data-provider-id' : 'data-patient-id';
    const rows = tbody.querySelectorAll('tr');
    let visibleCount = 0;

    rows.forEach(row => {
        const rowEntityId = row.getAttribute(dataAttr);
        const showRow = rowEntityId === entityId;
        row.style.display = showRow ? '' : 'none';
        if (showRow) visibleCount++;
    });

    if (visibleCount === 0) {
        const noDataRow = tbody.querySelector('.no-data-filtered');
        if (!noDataRow) {
            const row = document.createElement('tr');
            row.className = 'no-data-filtered';
            row.innerHTML = '<td colspan="5" class="no-data">No matching predictions</td>';
            tbody.appendChild(row);
        }
    } else {
        const noDataRow = tbody.querySelector('.no-data-filtered');
        if (noDataRow) noDataRow.remove();
    }

    showNotification(`Filtering by ${entityType}: ${entityId}`, 'info');
}

/**
 * Export currently visible rows in predictions table to CSV
 */
function exportFilteredCSV() {
    const tbody = document.getElementById('predictionsTableBody');
    if (!tbody) {
        showNotification('No table to export', 'error');
        return;
    }

    const rows = Array.from(tbody.querySelectorAll('tr')).filter(r => {
        if (r.classList.contains('no-data') || r.classList.contains('no-data-filtered')) return false;
        return r.style.display !== 'none';
    });

    if (rows.length === 0) {
        showNotification('No rows to export', 'warning');
        return;
    }

    // Build CSV header from table header in the same container
    const table = tbody.closest('table');
    const headers = Array.from(table.querySelectorAll('thead th')).map(h => h.textContent.trim());

    const csvRows = [headers.join(',')];

    rows.forEach(r => {
        const cols = Array.from(r.querySelectorAll('td')).map(td => {
            // Escape double quotes
            const text = td.textContent.replace(/"/g, '""').trim();
            return `"${text}"`;
        });
        csvRows.push(cols.join(','));
    });

    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `filtered_predictions_${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showNotification('Export started', 'success');
}

/**
 * Reset form to default values
 */
function resetForm() {
    document.getElementById('claimForm').reset();
    hideResults();
}

/**
 * Hide result cards
 */
function hideResults() {
    document.getElementById('resultCard').style.display = 'none';
    document.getElementById('explanationCard').style.display = 'none';
}

/**
 * Set loading state for submit button
 */
function setLoadingState(loading) {
    const btn = document.getElementById('submitBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');
    
    if (loading) {
        btn.disabled = true;
        btnText.style.display = 'none';
        btnLoading.style.display = 'flex';
    } else {
        btn.disabled = false;
        btnText.style.display = 'block';
        btnLoading.style.display = 'none';
    }
}

/**
 * Show notification message
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'error' ? '#fee2e2' : type === 'warning' ? '#fef3c7' : '#dbeafe'};
        color: ${type === 'error' ? '#991b1b' : type === 'warning' ? '#92400e' : '#1e40af'};
        border-radius: 0.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        z-index: 10000;
        max-width: 400px;
        animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}

// Add slide animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Close modal when clicking outside
const adminModalElement = document.getElementById('adminModal');
if (adminModalElement) {
    adminModalElement.addEventListener('click', (e) => {
        if (e.target.id === 'adminModal') {
            closeAdminDashboard();
        }
    });
}

// Export functions for global access
window.showAdminDashboard = showAdminDashboard;
window.closeAdminDashboard = closeAdminDashboard;
window.resetForm = resetForm;
window.applyFilters = applyFilters;
window.clearFilters = clearFilters;
window.filterByEntity = filterByEntity;

/**
 * Show model explanation tooltip
 */
function showTooltip(modelType) {
    const modal = document.getElementById('tooltipModal');
    const body = document.getElementById('tooltipBody');
    
    if (modelType === 'xgboost') {
        body.innerHTML = `
            <div class="tooltip-header">
                <div class="tooltip-icon xgboost">🎯</div>
                <div class="tooltip-title">
                    <h3>XGBoost Probability</h3>
                    <p>Supervised Machine Learning Model</p>
                </div>
            </div>
            
            <div class="tooltip-section">
                <h4>📚 What is it?</h4>
                <p>
                    XGBoost (eXtreme Gradient Boosting) is a supervised learning algorithm that was trained on thousands of 
                    labeled fraud cases. It learned the patterns and characteristics that distinguish fraudulent claims from genuine ones.
                </p>
            </div>
            
            <div class="tooltip-section">
                <h4>🎲 What does the percentage mean?</h4>
                <p>
                    The percentage represents the <strong>probability of fraud</strong> based on learned patterns:
                </p>
                <ul class="tooltip-list">
                    <li><strong>0-30%:</strong> Low risk - Claim characteristics match legitimate patterns</li>
                    <li><strong>30-70%:</strong> Medium risk - Mixed signals, requires investigation</li>
                    <li><strong>70-100%:</strong> High risk - Strong indicators of fraudulent behavior</li>
                </ul>
            </div>
            
            <div class="tooltip-section">
                <h4>💡 How it works:</h4>
                <p>
                    The model analyzes 13 features including claim amount, hospital stay, patient age, previous claims, 
                    and more. It compares these against thousands of known fraud and genuine cases to calculate the probability.
                </p>
            </div>
            
            <div class="tooltip-example">
                <strong>Example:</strong>
                If XGBoost shows 98.4%, it means based on all learned patterns, this claim has a 98.4% likelihood 
                of being fraudulent - very high confidence that it's fraud.
            </div>
        `;
    } else if (modelType === 'isoforest') {
        body.innerHTML = `
            <div class="tooltip-header">
                <div class="tooltip-icon isoforest">🌲</div>
                <div class="tooltip-title">
                    <h3>Isolation Forest Score</h3>
                    <p>Unsupervised Anomaly Detection</p>
                </div>
            </div>
            
            <div class="tooltip-section">
                <h4>📚 What is it?</h4>
                <p>
                    Isolation Forest is an anomaly detection algorithm that identifies <strong>unusual or outlier patterns</strong> 
                    without needing labeled data. It detects claims that deviate from normal behavior.
                </p>
            </div>
            
            <div class="tooltip-section">
                <h4>🎲 What does the percentage mean?</h4>
                <p>
                    The percentage represents how <strong>anomalous or unusual</strong> the claim is:
                </p>
                <ul class="tooltip-list">
                    <li><strong>0-30%:</strong> Normal - Claim follows typical patterns</li>
                    <li><strong>30-70%:</strong> Somewhat unusual - Has atypical characteristics</li>
                    <li><strong>70-100%:</strong> Highly anomalous - Very different from normal claims</li>
                </ul>
            </div>
            
            <div class="tooltip-section">
                <h4>💡 How it works:</h4>
                <p>
                    It isolates data points that are easier to separate from the rest. Claims that are unusual (potential fraud) 
                    are easier to isolate than normal claims, resulting in higher anomaly scores.
                </p>
            </div>
            
            <div class="tooltip-example">
                <strong>Example:</strong>
                If Isolation Forest shows 99.2%, it means this claim is extremely unusual compared to typical claims - 
                only 0.8% similar to normal patterns. This suggests it's an outlier that warrants investigation.
            </div>
            
            <div class="tooltip-section" style="background: #fef3c7; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                <h4>⚠️ Important Note:</h4>
                <p style="color: #92400e; margin: 0;">
                    High anomaly doesn't always mean fraud! A claim can be unusual for legitimate reasons 
                    (rare medical condition, emergency treatment, etc.). That's why we combine it with XGBoost for better accuracy.
                </p>
            </div>
        `;
    }
    
    modal.classList.add('active');
}

/**
 * Close tooltip modal
 */
function closeTooltip() {
    const modal = document.getElementById('tooltipModal');
    modal.classList.remove('active');
}

// Export tooltip functions
window.showTooltip = showTooltip;
window.closeTooltip = closeTooltip;

/**
 * Tab switching functionality
 */
function switchTab(tabName, evt) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName + 'Tab').classList.add('active');
    
    // If an event was provided, add active class to the originating button
    if (evt && evt.target) {
        evt.target.classList.add('active');
    }
    
    // Initialize network analysis when switching to network tab
    if (tabName === 'network' && typeof initNetworkAnalysis === 'function') {
        initNetworkAnalysis();
    }
}

/**
 * Batch input switching
 */
function switchBatchInput(inputType, evt) {
    // Hide all batch input contents
    document.querySelectorAll('.batch-input-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all batch tab buttons
    document.querySelectorAll('.batch-tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected input content
    document.getElementById(inputType + 'Input').classList.add('active');
    
    // Mark the originating button as active if we have an event object
    if (evt && evt.target) {
        evt.target.classList.add('active');
    }
}

/**
 * Handle CSV file upload
 */
document.getElementById('csvFile').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        
        fileName.textContent = `File: ${file.name}`;
        fileSize.textContent = `Size: ${(file.size / 1024).toFixed(1)} KB`;
        fileInfo.style.display = 'block';
        
        // Store file for processing
        window.batchFile = file;
    }
});

/**
 * Process batch claims
 */
async function processBatch() {
    const processBtn = document.getElementById('processBatchBtn');
    const btnText = processBtn.querySelector('.btn-text');
    const btnLoading = processBtn.querySelector('.btn-loading');
    
    // Show loading state
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline-flex';
    processBtn.disabled = true;
    
    try {
        let claimsData = [];
        
        // Get data based on active input
        const activeInput = document.querySelector('.batch-input-content.active');
        
        if (activeInput.id === 'fileInput') {
            // Process CSV file
            if (!window.batchFile) {
                throw new Error('Please select a CSV file first');
            }
            claimsData = await parseCSVFile(window.batchFile);
        } else {
            // Process JSON data
            const jsonText = document.getElementById('jsonData').value.trim();
            if (!jsonText) {
                throw new Error('Please enter JSON data');
            }
            claimsData = JSON.parse(jsonText);
        }
        
        if (!Array.isArray(claimsData) || claimsData.length === 0) {
            throw new Error('No valid claims data found');
        }
        
        if (claimsData.length > 1000) {
            throw new Error('Maximum 1000 claims allowed per batch');
        }
        
        // Keep the parsed claims for later 'Details' view
        window.batchInputClaims = claimsData;

        // Make batch prediction request
        const response = await fetch(`${API_BASE_URL}/batch-predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(claimsData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Batch processing failed');
        }
        
        const result = await response.json();
        
        // Display batch results
        displayBatchResults(result);
        
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
        console.error('Batch processing error:', error);
    } finally {
        // Hide loading state
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
        processBtn.disabled = false;
    }
}

/**
 * Parse CSV file to claims data
 */
async function parseCSVFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const csv = e.target.result;
                const lines = csv.split('\n').filter(line => line.trim());
                
                if (lines.length < 2) {
                    throw new Error('CSV file must have at least a header row and one data row');
                }
                
                const headers = lines[0].split(',').map(h => h.trim());
                const claims = [];
                
                for (let i = 1; i < lines.length; i++) {
                    const values = lines[i].split(',').map(v => v.trim());
                    if (values.length === headers.length) {
                        const claim = {};
                        headers.forEach((header, index) => {
                            const value = values[index];
                            // Convert to appropriate type
                            if (['age', 'gender', 'claim_amount', 'hospital_stay_days', 'previous_claims', 
                                 'treatment_type', 'provider_type', 'diagnosis_code', 'procedure_code', 
                                 'chronic_condition', 'insurance_type', 'policy_age_days', 'beneficiaries'].includes(header)) {
                                claim[header] = isNaN(value) ? value : (header === 'claim_amount' ? parseFloat(value) : parseInt(value));
                            }
                        });
                        claims.push(claim);
                    }
                }
                
                resolve(claims);
            } catch (error) {
                reject(new Error(`CSV parsing error: ${error.message}`));
            }
        };
        reader.onerror = () => reject(new Error('Failed to read CSV file'));
        reader.readAsText(file);
    });
}

/**
 * Display batch processing results
 */
function displayBatchResults(result) {
    const resultsDiv = document.getElementById('batchResults');
    const summary = result.summary;
    
    // Update summary stats
    document.getElementById('totalClaims').textContent = summary.total_claims;
    document.getElementById('processedClaims').textContent = summary.processed_claims;
    document.getElementById('failedClaims').textContent = summary.failed_claims;
    document.getElementById('fraudCount').textContent = summary.fraud_predictions;
    document.getElementById('batchFraudRate').textContent = `${summary.fraud_rate.toFixed(1)}%`;
    
    // Clear previous results
    const tableBody = document.getElementById('batchTableBody');
    tableBody.innerHTML = '';
    
    // Add results to table
    result.results.forEach((item, index) => {
        const row = document.createElement('tr');
        
        if (item.success) {
            row.innerHTML = `
                <td>${item.claim_index + 1}</td>
                <td><span class="status-success">✓ Processed</span></td>
                <td><span class="prediction-${item.final_prediction.toLowerCase()}">${item.final_prediction}</span></td>
                <td>${item.hybrid_risk_score.toFixed(3)}</td>
                <td>${(item.confidence * 100).toFixed(1)}%</td>
                <td>$${item.claim_amount?.toLocaleString() || 'N/A'}</td>
                <td>
                    <button class="btn btn-small" onclick="viewBatchDetails(${index})">Details</button>
                </td>
            `;
        } else {
            row.innerHTML = `
                <td>${item.claim_index + 1}</td>
                <td><span class="status-error">✗ Failed</span></td>
                <td colspan="4">${item.error}</td>
                <td></td>
            `;
        }
        
        tableBody.appendChild(row);
    });
    
    // Store results for export
    window.batchResults = result;
    
    // Show results
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * View batch claim details
 */
function viewBatchDetails(index) {
    const result = window.batchResults && window.batchResults.results && window.batchResults.results[index];
    if (!result) return;

    // Try to get original claim data from the parsed batch input (saved earlier)
    const claimData = (window.batchInputClaims && window.batchInputClaims[index]) || null;

    // Switch to single claim tab
    switchTab('single');

    // Populate form with claim data (prefer original claim fields)
    const form = document.getElementById('claimForm');
    const source = claimData || result;
    Object.keys(source).forEach(key => {
        const input = form.elements[key];
        if (!input) return;

        // Set value depending on input type
        if (input.tagName === 'INPUT') {
            if (input.type === 'number' || input.type === 'text' || input.type === 'email') {
                input.value = source[key];
            }
        } else if (input.tagName === 'SELECT') {
            input.value = source[key];
        } else if (input.tagName === 'TEXTAREA') {
            input.value = source[key];
        }
    });

    // Scroll to form
    form.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    showNotification('Claim data loaded. Click "Analyze Claim" to see detailed explanation.', 'info');
}

/**
 * Clear batch data
 */
function clearBatchData() {
    // Clear file input
    document.getElementById('csvFile').value = '';
    document.getElementById('fileInfo').style.display = 'none';
    window.batchFile = null;
    
    // Clear JSON input
    document.getElementById('jsonData').value = '';
    
    // Hide results
    document.getElementById('batchResults').style.display = 'none';
}

/**
 * Export batch results to CSV
 */
function exportBatchResults() {
    if (!window.batchResults) {
        showNotification('No results to export', 'error');
        return;
    }
    
    const results = window.batchResults.results;
    const csvContent = [
        // Header
        ['Claim_Index', 'Status', 'Claim_ID', 'Timestamp', 'Prediction', 'Risk_Score', 'Confidence', 
         'XGBoost_Probability', 'Isolation_Forest_Score', 'Claim_Amount', 'Error'].join(','),
        // Data
        ...results.map((item, index) => {
            if (item.success) {
                return [
                    item.claim_index + 1,
                    'Processed',
                    item.claim_id || '',
                    item.timestamp || '',
                    item.final_prediction,
                    item.hybrid_risk_score.toFixed(4),
                    item.confidence.toFixed(4),
                    item.xgboost_probability.toFixed(4),
                    item.isolation_forest_score.toFixed(4),
                    item.claim_amount || '',
                    ''
                ].join(',');
            } else {
                return [
                    item.claim_index + 1,
                    'Failed',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    `"${item.error}"`
                ].join(',');
            }
        })
    ].join('\n');
    
    // Create and download file
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `batch_results_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showNotification('Results exported successfully!', 'success');
}

// Export functions
window.switchTab = switchTab;
window.switchBatchInput = switchBatchInput;
window.processBatch = processBatch;
window.clearBatchData = clearBatchData;
window.viewBatchDetails = viewBatchDetails;
window.exportBatchResults = exportBatchResults;
window.clearHistory = clearHistory;
