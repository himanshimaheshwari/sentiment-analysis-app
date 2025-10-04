// Enhanced Full-Stack Sentiment Analysis Application with Direct Downloads
class FullStackSentimentApp {
    constructor() {
        this.isAnalyzing = false;
        this.currentResults = null;
        this.loadingSteps = [
            "Initializing sentiment engines...",
            "Loading TextBlob analyzer...",
            "Activating VADER sentiment model...",
            "Processing custom rule engine...",
            "Analyzing feedback patterns...",
            "Generating AI insights...",
            "Creating visualizations...",
            "Finalizing results..."
        ];
        this.currentStep = 0;

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#667eea';
            uploadArea.style.background = 'rgba(102, 126, 234, 0.1)';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'rgba(102, 126, 234, 0.3)';
            uploadArea.style.background = 'rgba(102, 126, 234, 0.05)';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'rgba(102, 126, 234, 0.3)';
            uploadArea.style.background = 'rgba(102, 126, 234, 0.05)';

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });
    }

    async handleFileUpload(file) {
        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.showError('Please upload a CSV file.');
            return;
        }

        if (file.size > 16 * 1024 * 1024) { // 16MB limit
            this.showError('File size must be less than 16MB.');
            return;
        }

        this.showProgress();
        this.showResults();
        this.startLoadingAnimation();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.stopLoadingAnimation();
                this.displayResults(result);
            } else {
                this.showError(result.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showError('Upload failed. Please try again.');
        }
    }

    showProgress() {
        document.getElementById('uploadProgress').style.display = 'block';
        let progress = 0;

        const interval = setInterval(() => {
            progress += Math.random() * 30;
            if (progress > 90) progress = 90;

            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = `Uploading... ${Math.round(progress)}%`;

            if (progress >= 90) {
                clearInterval(interval);
                document.getElementById('progressText').textContent = 'Processing...';
            }
        }, 200);
    }

    showResults() {
        document.getElementById('results').style.display = 'block';
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
    }

    startLoadingAnimation() {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('dashboard').style.display = 'none';

        this.currentStep = 0;
        const loadingText = document.getElementById('loadingStep');

        const stepInterval = setInterval(() => {
            if (this.currentStep < this.loadingSteps.length) {
                loadingText.textContent = this.loadingSteps[this.currentStep];
                this.currentStep++;
            } else {
                loadingText.textContent = "Almost done...";
            }
        }, 800);

        // Store interval for cleanup
        this.loadingInterval = stepInterval;
    }

    stopLoadingAnimation() {
        if (this.loadingInterval) {
            clearInterval(this.loadingInterval);
        }

        setTimeout(() => {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('dashboard').style.display = 'block';
            document.getElementById('uploadProgress').style.display = 'none';
        }, 1000);
    }

    displayResults(results) {
        this.currentResults = results;

        // Update stats
        document.getElementById('totalFeedback').textContent = results.stats.total_feedback;
        document.getElementById('positivePercent').textContent = results.stats.positive_percent + '%';
        document.getElementById('averageScore').textContent = results.stats.average_polarity;
        document.getElementById('subjectivityScore').textContent = results.stats.average_subjectivity;

        // Display charts
        this.displayCharts(results.charts);

        // Display insights
        this.displayInsights(results.insights);

        // Show download buttons
        this.setupDownloadButtons();
    }

    setupDownloadButtons() {
        const downloadChartsBtn = document.getElementById('downloadChartsBtn');
        const downloadInsightsBtn = document.getElementById('downloadInsightsBtn');
        const downloadDatasetBtn = document.getElementById('downloadDatasetBtn');

        downloadChartsBtn.style.display = 'inline-flex';
        downloadInsightsBtn.style.display = 'inline-flex';
        downloadDatasetBtn.style.display = 'inline-flex';

        downloadChartsBtn.onclick = () => this.downloadCharts();
        downloadInsightsBtn.onclick = () => this.downloadInsights();
        downloadDatasetBtn.onclick = () => this.downloadDataset();
    }

    downloadCharts() {
        this.triggerDownload('/download-charts', 'Charts download started! Check your Downloads folder.');
    }

    downloadInsights() {
        this.triggerDownload('/download-insights', 'Insights download started! Check your Downloads folder.');
    }

    downloadDataset() {
        this.triggerDownload('/download-dataset', 'Dataset download started! Check your Downloads folder.');
    }

    triggerDownload(url, successMessage) {
        try {
            // Create temporary link and trigger download
            const link = document.createElement('a');
            link.href = url;
            link.style.display = 'none';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Show success message
            this.showSuccessMessage(successMessage);
        } catch (error) {
            console.error('Download error:', error);
            this.showError('Download failed. Please try again.');
        }
    }

    showSuccessMessage(message) {
        const successDiv = document.createElement('div');
        successDiv.className = 'download-notification success';
        successDiv.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span>${message}</span>
            <button class="close-btn" onclick="this.parentElement.remove()">×</button>
        `;

        document.body.appendChild(successDiv);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (document.body.contains(successDiv)) {
                successDiv.classList.add('fade-out');
                setTimeout(() => {
                    if (document.body.contains(successDiv)) {
                        document.body.removeChild(successDiv);
                    }
                }, 300);
            }
        }, 5000);
    }

    displayCharts(charts) {
        const chartsGrid = document.getElementById('chartsGrid');
        chartsGrid.innerHTML = '';

        const chartTitles = {
            'sentiment_pie': 'Sentiment Distribution',
            'category_bar': 'Category Performance',
            'polarity_hist': 'Polarity Distribution',
            'word_freq': 'Word Frequency Analysis',
            'comparison': 'Method Comparison'
        };

        Object.entries(charts).forEach(([key, base64Data]) => {
            const chartItem = document.createElement('div');
            chartItem.className = 'chart-item';

            chartItem.innerHTML = `
                <h4><i class="fas fa-chart-bar"></i> ${chartTitles[key] || key}</h4>
                <img src="data:image/png;base64,${base64Data}" alt="${chartTitles[key] || key}" />
            `;

            chartsGrid.appendChild(chartItem);
        });
    }

    displayInsights(insights) {
        const insightsGrid = document.getElementById('insightsGrid');
        insightsGrid.innerHTML = '';

        insights.forEach((insight, index) => {
            const insightCard = document.createElement('div');
            insightCard.className = `insight-card ${insight.priority}`;

            const trendIcon = insight.trend === 'up' ? '📈' : insight.trend === 'down' ? '📉' : '📊';

            insightCard.innerHTML = `
                <div class="insight-header">
                    <div class="insight-icon">${insight.icon}</div>
                    <div>
                        <div class="insight-title">${insight.title}</div>
                    </div>
                </div>
                <div class="insight-text">${insight.text}</div>
                <div class="insight-meta">
                    <span class="insight-metric">${insight.metric}</span>
                    <span class="insight-trend">${trendIcon} ${insight.trend}</span>
                </div>
            `;

            // Add animation delay
            insightCard.style.animationDelay = `${index * 0.1}s`;
            insightCard.style.animation = 'fadeInUp 0.6s ease-out forwards';

            insightsGrid.appendChild(insightCard);
        });
    }

    showError(message) {
        // Hide loading and progress
        document.getElementById('loading').style.display = 'none';
        document.getElementById('uploadProgress').style.display = 'none';

        if (this.loadingInterval) {
            clearInterval(this.loadingInterval);
        }

        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'download-notification error';
        errorDiv.innerHTML = `
            <i class="fas fa-exclamation-circle"></i>
            <span>${message}</span>
            <button class="close-btn" onclick="this.parentElement.remove()">×</button>
        `;

        document.body.appendChild(errorDiv);

        // Auto-remove after 7 seconds
        setTimeout(() => {
            if (document.body.contains(errorDiv)) {
                errorDiv.classList.add('fade-out');
                setTimeout(() => {
                    if (document.body.contains(errorDiv)) {
                        document.body.removeChild(errorDiv);
                    }
                }, 300);
            }
        }, 7000);

        // Reset file input
        document.getElementById('fileInput').value = '';
    }
}

// Demo functionality
async function runDemo() {
    const app = window.sentimentApp;

    app.showResults();
    app.startLoadingAnimation();

    try {
        const response = await fetch('/demo');
        const result = await response.json();

        if (result.success) {
            app.stopLoadingAnimation();
            app.displayResults(result);
        } else {
            app.showError(result.error || 'Demo failed');
        }
    } catch (error) {
        console.error('Demo error:', error);
        app.showError('Demo failed. Please try again.');
    }
}

// Navigation functions
function scrollToUpload() {
    document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
}

function exportCharts() {
    // Use the new download functionality
    if (window.sentimentApp && window.sentimentApp.currentResults) {
        window.sentimentApp.downloadCharts();
    } else {
        alert('No charts available for download. Please run an analysis first.');
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    window.sentimentApp = new FullStackSentimentApp();

    // Add CSS for notifications
    const notificationStyle = document.createElement('style');
    notificationStyle.textContent = `
        .download-notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 10000;
            display: flex;
            align-items: center;
            gap: 10px;
            max-width: 400px;
            font-weight: 600;
            border-left: 4px solid;
            animation: slideIn 0.3s ease-out;
        }

        .download-notification.success {
            border-left-color: #28a745;
            color: #155724;
        }

        .download-notification.error {
            border-left-color: #dc3545;
            color: #721c24;
        }

        .download-notification i {
            font-size: 1.2em;
        }

        .download-notification.success i {
            color: #28a745;
        }

        .download-notification.error i {
            color: #dc3545;
        }

        .download-notification .close-btn {
            background: none;
            border: none;
            font-size: 1.5em;
            cursor: pointer;
            color: #666;
            margin-left: auto;
            padding: 0;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .download-notification .close-btn:hover {
            color: #333;
        }

        .download-notification.fade-out {
            animation: slideOut 0.3s ease-out forwards;
        }

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
    document.head.appendChild(notificationStyle);

    // Smooth scrolling for navigation
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

    // Header scroll effect
    const header = document.querySelector('.header');
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 100) {
            header.style.background = 'rgba(255, 255, 255, 0.98)';
            header.style.backdropFilter = 'blur(20px)';
        } else {
            header.style.background = 'rgba(255, 255, 255, 0.95)';
            header.style.backdropFilter = 'blur(20px)';
        }
    });

    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe feature cards for animations
    document.querySelectorAll('.feature-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });

    console.log('🚀 Enhanced Sentiment Analysis App Initialized');
    console.log('📊 Ready for file uploads, demo analysis, and downloads');
    console.log('📁 Chart, insights, and dataset downloads enabled');
});

// Easter egg functionality
let clickCount = 0;
document.addEventListener('DOMContentLoaded', function() {
    const aiIcon = document.querySelector('.ai-icon');
    if (aiIcon) {
        aiIcon.addEventListener('click', () => {
            clickCount++;
            if (clickCount === 5) {
                const msg = '🎉 You found the easter egg! This app now has enhanced download capabilities! 🤖📊';
                window.sentimentApp.showSuccessMessage(msg);
                clickCount = 0;
            }
        });
    }
});