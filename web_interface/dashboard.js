/**
 * NeuroLM Video Experiment Dashboard
 * Real-time visualization of EEG metrics and video synchronization
 */

class NeuroLMDashboard {
    constructor() {
        this.websocket = null;
        this.videoMetrics = {};
        this.experimentRunning = false;
        this.startTime = null;
        this.currentVideoUrl = null;
        this.currentVideoTitle = null;
        this.videoColors = {};
        this.colorPalette = [
            '#4A90E2', // Blue
            '#2ECC71', // Green  
            '#E74C3C', // Red
            '#F39C12', // Orange
            '#9B59B6', // Purple
            '#1ABC9C', // Turquoise
            '#E67E22', // Dark Orange
            '#34495E', // Dark Blue Grey
            '#F1C40F', // Yellow
            '#E91E63', // Pink
            '#00BCD4', // Cyan
            '#FF5722', // Deep Orange
            '#795548', // Brown
            '#607D8B', // Blue Grey
            '#CDDC39'  // Lime
        ];
        this.nextColorIndex = 0;
        this.metricsData = [];
        this.eegData = [];
        this.quadrantData = [];
        this.startTime = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.connectWebSocket();
        this.updateTimestamp();
        
        // Update timestamp every second
        setInterval(() => this.updateTimestamp(), 1000);
    }
    
    setupEventListeners() {
        // YouTube video loading
        document.getElementById('loadYouTubeBtn').addEventListener('click', () => this.loadYouTubeVideo());
        document.getElementById('youtubeUrl').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.loadYouTubeVideo();
        });
        
        // Experiment controls
        document.getElementById('startExperiment').addEventListener('click', () => this.startExperiment());
        document.getElementById('stopExperiment').addEventListener('click', () => this.stopExperiment());
        
        // Export data
        document.getElementById('export-data').addEventListener('click', () => this.exportData());
        
        // EEG source change
        document.getElementById('eeg-source').addEventListener('change', (e) => this.changeEEGSource(e.target.value));
    }
    
    initializeCharts() {
        // 2D Metrics Quadrant Chart (Scatter Plot)
        const quadrantCtx = document.getElementById('metricsQuadrantChart').getContext('2d');
        this.metricsQuadrantChart = new Chart(quadrantCtx, {
            type: 'scatter',
            data: {
                datasets: []  // Will be populated dynamically for each video
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#e2e8f0',
                            font: { size: 12 }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(30, 41, 59, 0.9)',
                        titleColor: '#e2e8f0',
                        bodyColor: '#e2e8f0',
                        borderColor: '#475569',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                const datasetLabel = context.dataset.label;
                                return [
                                    `Video: ${datasetLabel}`,
                                    `Attention: ${context.parsed.x.toFixed(3)}`,
                                    `Engagement: ${context.parsed.y.toFixed(3)}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Attention →',
                            color: '#e2e8f0',
                            font: { size: 14, weight: 'bold' }
                        },
                        ticks: {
                            color: '#94a3b8',
                            stepSize: 0.2
                        },
                        grid: {
                            color: 'rgba(148, 163, 184, 0.1)'
                        }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Engagement ↑',
                            color: '#e2e8f0',
                            font: { size: 14, weight: 'bold' }
                        },
                        ticks: {
                            color: '#94a3b8',
                            stepSize: 0.2
                        },
                        grid: {
                            color: 'rgba(148, 163, 184, 0.1)'
                        }
                    }
                },
                elements: {
                    line: {
                        tension: 0  // No curve between points
                    }
                }
            }
        });
        
        // Metrics Timeline Chart
        const metricsTimelineCtx = document.getElementById('metricsTimelineChart').getContext('2d');
        this.metricsTimelineChart = new Chart(metricsTimelineCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Fp1',
                        data: [],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.2)',
                        tension: 0.2,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'Fp2',
                        data: [],
                        borderColor: '#f97316',
                        backgroundColor: 'rgba(249, 115, 22, 0.2)',
                        tension: 0.2,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'C3',
                        data: [],
                        borderColor: '#eab308',
                        backgroundColor: 'rgba(234, 179, 8, 0.2)',
                        tension: 0.2,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'C4',
                        data: [],
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.2)',
                        tension: 0.2,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'P7',
                        data: [],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.2)',
                        tension: 0.2,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'P8',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.2)',
                        tension: 0.2,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'O1',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.2)',
                        tension: 0.2,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'O2',
                        data: [],
                        borderColor: '#ec4899',
                        backgroundColor: 'rgba(236, 72, 153, 0.2)',
                        tension: 0.2,
                        pointRadius: 0,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#f8fafc'
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'second',
                            displayFormats: {
                                second: 'mm:ss'
                            }
                        },
                        ticks: {
                            color: '#cbd5e1'
                        },
                        grid: {
                            color: '#334155'
                        }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        ticks: {
                            color: '#cbd5e1'
                        },
                        grid: {
                            color: '#334155'
                        }
                    }
                },
                animation: {
                    duration: 0
                }
            }
        });
        
        // EEG Channels Chart (8 channels: Fp1, Fp2, C3, C4, P7, P8, O1, O2)
        const eegCtx = document.getElementById('eegChannelsChart').getContext('2d');
        this.eegChannelsChart = new Chart(eegCtx, {
            type: 'line',
            data: {
                datasets: [
                    { label: 'Fp1', data: [], borderColor: '#ef4444', tension: 0.1 },
                    { label: 'Fp2', data: [], borderColor: '#f97316', tension: 0.1 },
                    { label: 'C3', data: [], borderColor: '#eab308', tension: 0.1 },
                    { label: 'C4', data: [], borderColor: '#22c55e', tension: 0.1 },
                    { label: 'P7', data: [], borderColor: '#06b6d4', tension: 0.1 },
                    { label: 'P8', data: [], borderColor: '#3b82f6', tension: 0.1 },
                    { label: 'O1', data: [], borderColor: '#8b5cf6', tension: 0.1 },
                    { label: 'O2', data: [], borderColor: '#ec4899', tension: 0.1 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#f8fafc',
                            font: {
                                size: 10
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'second',
                            displayFormats: {
                                second: 'mm:ss'
                            }
                        },
                        ticks: {
                            color: '#cbd5e1'
                        },
                        grid: {
                            color: '#334155'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#cbd5e1'
                        },
                        grid: {
                            color: '#334155'
                        }
                    }
                },
                animation: {
                    duration: 0
                }
            }
        });
    }
    
    connectWebSocket() {
        try {
            console.log('🔗 Attempting to connect to WebSocket at ws://localhost:8765');
            this.websocket = new WebSocket('ws://localhost:8765');
            
            this.websocket.onopen = () => {
                console.log('✅ WebSocket connection opened successfully');
                this.isConnected = true;
                this.updateConnectionStatus();
                this.addLogEntry('WebSocket connected');
                
                // Send a test message to verify connection
                console.log('📤 Sending test message to WebSocket');
                this.websocket.send(JSON.stringify({
                    command: 'get_status'
                }));
                
                // No auto-start - wait for manual user action
                console.log('⏳ WebSocket ready. Waiting for manual experiment start...');
            };
            
            this.websocket.onmessage = (event) => {
                console.log('📨 Received WebSocket message:', event.data);
                try {
                    const data = JSON.parse(event.data);
                    console.log('📊 Parsed data:', data);
                    this.handleWebSocketMessage(data);
                } catch (e) {
                    console.error('❌ Error parsing WebSocket message:', e);
                }
            };
            
            this.websocket.onclose = () => {
                this.isConnected = false;
                this.updateConnectionStatus();
                this.addLogEntry('WebSocket disconnected');
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => this.connectWebSocket(), 3000);
            };
            
            this.websocket.onerror = (error) => {
                this.addLogEntry(`WebSocket error: ${error.message}`);
            };
            
        } catch (error) {
            this.addLogEntry(`Failed to connect WebSocket: ${error.message}`);
        }
    }
    
    updateMetricCard(metricName, value, level) {
        // Map metric names to HTML element IDs
        const metricIds = {
            'attention': 'attentionValue',
            'engagement': 'engagementValue', 
            'workload': 'workloadValue',
            'alpha_theta': 'alphaTheta'
        };
        
        // Update numeric value using direct ID
        const elementId = metricIds[metricName];
        if (elementId) {
            const valueElement = document.getElementById(elementId);
            if (valueElement) {
                valueElement.textContent = value.toFixed(3);
                console.log(`✅ Updated ${metricName}: ${value.toFixed(3)} (${level})`);
            } else {
                console.log(`❌ Element not found: ${elementId}`);
            }
        } else {
            console.log(`❌ Unknown metric: ${metricName}`);
        }
    }
    
    storeVideoMetrics(data) {
        // Store metrics for video comparison
        if (this.currentVideoUrl && data.attention !== undefined) {
            const videoId = this.extractVideoId(this.currentVideoUrl);
            if (!this.videoMetrics[videoId]) {
                this.videoMetrics[videoId] = {
                    url: this.currentVideoUrl,
                    title: this.currentVideoTitle || 'Unknown Video',
                    metrics: []
                };
            }
            
            this.videoMetrics[videoId].metrics.push({
                timestamp: new Date(),
                attention: data.attention,
                engagement: data.engagement,
                workload: data.workload,
                alpha_theta: data.alpha_theta_ratio
            });
            
            console.log(`💾 Stored metrics for video ${videoId}`);
        }
    }
    
    extractVideoId(url) {
        // Extract YouTube video ID from URL
        const match = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/);
        return match ? match[1] : 'unknown';
    }
    
    handleWebSocketMessage(data) {
        console.log('🎯 Handling WebSocket message:', data);
        const timestamp = new Date();
        
        // Check message type
        if (data.type) {
            console.log(`📋 Message type: ${data.type}`);
            
            if (data.type === 'metrics_update') {
                console.log('📊 Processing metrics update:', {
                    attention: data.attention,
                    engagement: data.engagement,
                    workload: data.workload
                });
            }
        }
        
        // Update real-time metrics
        if (data.attention !== undefined) {
            console.log(`🧠 Updating attention: ${data.attention}`);
            this.updateMetricCard('attention', data.attention, data.attention_level || 'Medium');
        }
        
        if (data.engagement !== undefined) {
            console.log(`🎯 Updating engagement: ${data.engagement}`);
            this.updateMetricCard('engagement', data.engagement, data.engagement_level || 'Medium');
        }
        
        if (data.workload !== undefined) {
            console.log(`⚡ Updating workload: ${data.workload}`);
            this.updateMetricCard('workload', data.workload, data.workload_level || 'Low');
        }
        
        // Store metrics for comparison
        this.storeVideoMetrics(data);
        
        // Update 2D quadrant chart (works with or without video)
        if (data.attention !== undefined && data.engagement !== undefined) {
            console.log(`📊 Adding point to quadrant: (${data.attention}, ${data.engagement})`);
            const videoId = this.currentVideoUrl ? this.extractVideoId(this.currentVideoUrl) : 'simulator';
            const videoTitle = this.currentVideoTitle || 'Real-time EEG Stream';
            this.addQuadrantPoint(data.attention, data.engagement, videoId, videoTitle, true);
        }
        
        // Add to metrics timeline chart
        if (data.attention !== undefined || data.engagement !== undefined || data.workload !== undefined) {
            console.log('📈 Adding to timeline chart');
            this.addMetricsDataPoint(timestamp, data);
        }
        
        // Add to EEG chart if raw data is available
        if (data.eeg_channels) {
            console.log(`🧠 Adding EEG data: ${data.eeg_channels.length} channels`);
            this.addEEGDataPoint(timestamp, data.eeg_channels);
        }
    }
    
    addEEGDataPoint(timestamp, channels) {
        // Add new data points for each channel
        channels.forEach((value, index) => {
            if (index < this.eegChannelsChart.data.datasets.length) {
                this.eegChannelsChart.data.datasets[index].data.push({
                    x: timestamp,
                    y: value
                });
            }
        });
        
        // Keep only last 10 seconds of EEG data
        const cutoffTime = new Date(timestamp.getTime() - 10000);
        this.eegChannelsChart.data.datasets.forEach(dataset => {
            dataset.data = dataset.data.filter(point => point.x > cutoffTime);
        });
        
        this.eegChannelsChart.update('none');
    }
    
    // Get or create dataset for a video
    getVideoDataset(videoId, videoTitle, isCurrentVideo = false) {
        // Check if dataset already exists
        let dataset = this.metricsQuadrantChart.data.datasets.find(ds => ds.videoId === videoId);
        
        if (!dataset) {
            // Get unique color for this video
            if (!this.videoColors[videoId]) {
                this.videoColors[videoId] = this.colorPalette[this.nextColorIndex % this.colorPalette.length];
                this.nextColorIndex++;
            }
            
            const color = this.videoColors[videoId];
            const shortTitle = videoTitle.length > 20 ? videoTitle.substring(0, 20) + '...' : videoTitle;
            
            // Create new dataset for this video
            dataset = {
                label: `${videoId} - ${shortTitle}`,
                videoId: videoId,
                data: [],
                backgroundColor: color + '80', // Add transparency
                borderColor: color,
                borderWidth: 2,
                pointRadius: isCurrentVideo ? 8 : 6,
                pointHoverRadius: isCurrentVideo ? 12 : 10,
                showLine: false
            };
            
            this.metricsQuadrantChart.data.datasets.push(dataset);
        }
        
        // Update point size if this is current video
        if (isCurrentVideo) {
            dataset.pointRadius = 8;
            dataset.pointHoverRadius = 12;
        }
        
        return dataset;
    }
    
    // Add point to 2D quadrant chart
    addQuadrantPoint(attention, engagement, videoId = null, videoTitle = 'Unknown', isCurrentVideo = true) {
        if (!videoId && this.currentVideoUrl) {
            videoId = this.extractVideoId(this.currentVideoUrl);
            videoTitle = this.currentVideoTitle || 'Unknown Video';
        }
        
        // Use default ID if no video (for simulator mode)
        if (!videoId) {
            videoId = 'simulator';
            videoTitle = 'Real-time EEG Stream';
        }
        
        // Get or create dataset for this video
        const dataset = this.getVideoDataset(videoId, videoTitle, isCurrentVideo);
        
        // Add new discrete point to the dataset with QAM trail effect
        dataset.data.push({
            x: attention,
            y: engagement,
            timestamp: new Date() // Add timestamp for trail effect
        });
        
        // Keep last 50 points for QAM "manchado" effect (trail)
        if (dataset.data.length > 50) {
            dataset.data.shift();
        }
        
        // Update point opacity based on age (newer points more opaque)
        dataset.data.forEach((point, index) => {
            const age = dataset.data.length - index;
            const opacity = Math.max(0.1, 1 - (age / 50)); // Fade older points
            if (!dataset.pointBackgroundColor) dataset.pointBackgroundColor = [];
            if (!dataset.pointBorderColor) dataset.pointBorderColor = [];
            
            const baseColor = dataset.backgroundColor || '#3b82f6';
            const rgb = this.hexToRgb(baseColor);
            dataset.pointBackgroundColor[index] = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${opacity})`;
            dataset.pointBorderColor[index] = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${Math.min(1, opacity + 0.3)})`;
        });
        
        console.log(`🔵 Added QAM point: (${attention.toFixed(3)}, ${engagement.toFixed(3)}) - ${dataset.data.length} points total`);
        this.metricsQuadrantChart.update('none');
    }
    
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : {r: 59, g: 130, b: 246}; // Default blue
    }
    
    addMetricsDataPoint(timestamp, data) {
        // Add new data points to timeline chart
        this.metricsTimelineChart.data.datasets[0].data.push({
            x: timestamp,
            y: data.attention || 0
        });
        this.metricsTimelineChart.data.datasets[1].data.push({
            x: timestamp,
            y: data.engagement || 0
        });
        this.metricsTimelineChart.data.datasets[2].data.push({
            x: timestamp,
            y: data.workload || 0
        });
        
        // Keep only last 30 seconds of data
        const cutoffTime = new Date(timestamp.getTime() - 30000);
        this.metricsTimelineChart.data.datasets.forEach(dataset => {
            dataset.data = dataset.data.filter(point => point.x > cutoffTime);
        });
        
        this.metricsTimelineChart.update('none');
    }
    
    startExperiment() {
        // Get current selections
        const eegSource = document.getElementById('eeg-source').value;
        const segmentDuration = parseInt(document.getElementById('segment-duration').value);
        const youtubeVideo = document.getElementById('youtubeVideo');
        const hasVideo = youtubeVideo.src && youtubeVideo.src.length > 0;
        
        console.log(`🎮 Start Experiment requested - EEG: ${eegSource}, Segment: ${segmentDuration}s, Video: ${hasVideo}`);
        
        // Validation logic based on EEG source
        if (eegSource === 'simulator') {
            // Simulator mode: Can work with or without video
            console.log('🤖 Simulator mode: Starting experiment...');
        } else if (eegSource === 'lsl') {
            // LSL mode: Requires video for proper experiment
            if (!hasVideo) {
                alert('⚠️ LSL Stream mode requires a video to be loaded first. Please load a YouTube video.');
                console.log('❌ LSL mode requires video - experiment not started');
                return;
            }
            console.log('📡 LSL Stream mode: Starting experiment with video...');
        } else {
            alert('⚠️ Please select a valid EEG source (LSL Stream or Simulator)');
            console.log('❌ Invalid EEG source - experiment not started');
            return;
        }
        
        // Check WebSocket connection
        if (!this.isConnected) {
            alert('⚠️ WebSocket not connected. Please wait for connection or refresh the page.');
            console.log('❌ WebSocket not connected - experiment not started');
            return;
        }
        
        // Start experiment if all validations pass
        if (!this.experimentRunning) {
            this.experimentRunning = true;
            this.startTime = new Date();
            
            // Start video if available
            if (hasVideo) {
                youtubeVideo.play();
                console.log('🎬 Video started');
            }
            
            this.addLogEntry(`Experiment started - ${eegSource} mode, ${segmentDuration}s segments`);
            
            // Send start signal via WebSocket with proper parameters
            this.websocket.send(JSON.stringify({
                command: 'start_experiment',
                eeg_source: eegSource,
                segment_duration: segmentDuration,
                has_video: hasVideo
            }));
            
            console.log(`✅ Experiment started successfully - ${eegSource} mode, ${segmentDuration}s segments`);
        } else {
            console.log('⚠️ Experiment already running');
        }
    }
    
    pauseExperiment() {
        if (this.experimentRunning) {
            const video = document.getElementById('experiment-video');
            video.pause();
            
            this.addLogEntry('Experiment paused');
            
            if (this.isConnected) {
                this.websocket.send(JSON.stringify({
                    command: 'pause_experiment'
                }));
            }
        }
    }
    
    stopExperiment() {
        this.experimentRunning = false;
        
        const video = document.getElementById('experiment-video');
        video.pause();
        video.currentTime = 0;
        
        this.addLogEntry('Experiment stopped');
        
        if (this.isConnected) {
            this.websocket.send(JSON.stringify({
                command: 'stop_experiment'
            }));
        }
    }
    
    loadYouTubeVideo() {
        const url = document.getElementById('youtubeUrl').value.trim();
        if (!url) {
            this.addLogEntry('❌ Please enter a YouTube URL');
            return;
        }
        
        // Extract video ID from YouTube URL
        const videoId = this.extractYouTubeVideoId(url);
        if (!videoId) {
            this.addLogEntry('❌ Invalid YouTube URL');
            return;
        }
        
        // Hide regular video and placeholder
        document.getElementById('experimentVideo').style.display = 'none';
        document.getElementById('videoPlaceholder').style.display = 'none';
        
        // Show and configure YouTube iframe
        const youtubeVideo = document.getElementById('youtubeVideo');
        youtubeVideo.src = `https://www.youtube.com/embed/${videoId}?enablejsapi=1&autoplay=0`;
        youtubeVideo.style.display = 'block';
        youtubeVideo.style.width = '100%';
        youtubeVideo.style.height = '100%';
        
        // Update current video display
        const videoTitle = this.getVideoTitleFromUrl(url);
        document.getElementById('currentVideo').textContent = videoTitle;
        
        this.addLogEntry(`📺 YouTube video loaded: ${videoTitle}`);
    }
    
    extractYouTubeVideoId(url) {
        const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
        const match = url.match(regex);
        return match ? match[1] : null;
    }
    
    getVideoTitleFromUrl(url) {
        // Extract a simple title from URL or use default
        const videoId = this.extractYouTubeVideoId(url);
        return videoId ? `YouTube Video (${videoId})` : 'YouTube Video';
    }
    
    storeVideoMetrics(data) {
        const currentVideoElement = document.getElementById('currentVideo');
        const videoName = currentVideoElement.textContent;
        
        if (videoName && videoName !== 'No video loaded') {
            // Add to comparison list
            const comparisonList = document.getElementById('videoComparisonList');
            
            // Check if this video already exists
            const existingItems = comparisonList.querySelectorAll('.comparison-item');
            let found = false;
            
            existingItems.forEach(item => {
                const name = item.querySelector('.video-name');
                if (name && name.textContent === videoName) {
                    // Update existing metrics
                    this.updateComparisonItem(item, data);
                    found = true;
                }
            });
            
            if (!found) {
                // Create new comparison item
                this.createComparisonItem(comparisonList, videoName, data);
            }
        }
    }
    
    createComparisonItem(container, videoName, data) {
        // Remove "no videos" placeholder if it exists
        const placeholder = container.querySelector('.comparison-item');
        if (placeholder && placeholder.querySelector('.video-name').textContent === 'No videos tested yet') {
            placeholder.remove();
        }
        
        const item = document.createElement('div');
        item.className = 'comparison-item';
        item.innerHTML = `
            <span class="video-name">${videoName}</span>
            <div class="video-metrics">
                <span class="metric-summary">A:${data.attention.toFixed(2)}</span>
                <span class="metric-summary">E:${data.engagement.toFixed(2)}</span>
                <span class="metric-summary">W:${data.workload.toFixed(2)}</span>
                <span class="metric-summary">α/θ:${data.alpha_theta_ratio.toFixed(2)}</span>
            </div>
        `;
        container.appendChild(item);
    }
    
    updateComparisonItem(item, data) {
        const metricsDiv = item.querySelector('.video-metrics');
        if (metricsDiv) {
            metricsDiv.innerHTML = `
                <span class="metric-summary">A:${data.attention.toFixed(2)}</span>
                <span class="metric-summary">E:${data.engagement.toFixed(2)}</span>
                <span class="metric-summary">W:${data.workload.toFixed(2)}</span>
                <span class="metric-summary">α/θ:${data.alpha_theta_ratio.toFixed(2)}</span>
            `;
        }
    }
    
    changeEEGSource(source) {
        this.addLogEntry(`EEG source changed to: ${source}`);
        
        if (this.isConnected) {
            this.websocket.send(JSON.stringify({
                command: 'change_eeg_source',
                source: source
            }));
        }
    }
    
    exportData() {
        const data = {
            experiment_info: {
                start_time: this.startTime,
                duration: this.experimentRunning ? new Date() - this.startTime : 0,
                segment_duration: document.getElementById('segment-duration').value
            },
            metrics_data: this.metricsData,
            eeg_data: this.eegData
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `neurolm_experiment_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.addLogEntry('Data exported');
    }
    
    updateConnectionStatus() {
        const websocketStatus = document.getElementById('websocket-status');
        const eegStatus = document.getElementById('eeg-status');
        const neurolmStatus = document.getElementById('neurolm-status');
        
        if (this.isConnected) {
            websocketStatus.textContent = 'WebSocket: Connected';
            eegStatus.classList.add('connected');
        } else {
            websocketStatus.textContent = 'WebSocket: Disconnected';
            eegStatus.classList.remove('connected');
            neurolmStatus.classList.remove('connected');
        }
    }
    
    addLogEntry(message) {
        const logContent = document.getElementById('experiment-log');
        const timestamp = new Date().toLocaleTimeString();
        
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `
            <span class="timestamp">${timestamp}</span>
            <span class="message">${message}</span>
        `;
        
        logContent.appendChild(entry);
        logContent.scrollTop = logContent.scrollHeight;
        
        // Keep only last 50 log entries
        while (logContent.children.length > 50) {
            logContent.removeChild(logContent.firstChild);
        }
    }
    
    updateTimestamp() {
        if (this.experimentRunning && this.startTime) {
            const elapsed = new Date() - this.startTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            
            const timestampElement = document.getElementById('experiment-timestamp');
            if (timestampElement) {
                timestampElement.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }
    }
    
    storeVideoMetrics(data) {
        if (this.currentVideoUrl) {
            const videoId = this.extractVideoId(this.currentVideoUrl);
            if (!this.videoMetrics[videoId]) {
                this.videoMetrics[videoId] = {
                    url: this.currentVideoUrl,
                    title: this.currentVideoTitle || 'Unknown Video',
                    metrics: []
                };
            }
            
            this.videoMetrics[videoId].metrics.push({
                timestamp: new Date(),
                attention: data.attention,
                engagement: data.engagement,
                workload: data.workload,
                alpha_theta: data.alpha_theta_ratio
            });
            
            // Update comparison panel
            this.updateVideoComparison();
        }
    }
    
    updateVideoComparison() {
        const comparisonContent = document.getElementById('video-comparison-content');
        if (!comparisonContent) return;
        
        comparisonContent.innerHTML = '';
        
        Object.values(this.videoMetrics).forEach(video => {
            if (video.metrics.length > 0) {
                const avgMetrics = this.calculateAverageMetrics(video.metrics);
                const videoId = this.extractVideoId(video.url);
                const videoColor = this.videoColors[videoId] || '#4A90E2';
                
                const videoItem = document.createElement('div');
                videoItem.className = 'video-comparison-item';
                videoItem.style.borderLeftColor = videoColor;
                videoItem.innerHTML = `
                    <div class="video-id">
                        <span class="color-indicator" style="background-color: ${videoColor}"></span>
                        ID: ${videoId}
                    </div>
                    <div class="video-title">${video.title}</div>
                    <div class="video-metrics">
                        <span>📊 Attention: ${avgMetrics.attention.toFixed(3)}</span>
                        <span>🎯 Engagement: ${avgMetrics.engagement.toFixed(3)}</span>
                        <span>⚡ Workload: ${avgMetrics.workload.toFixed(3)}</span>
                    </div>
                `;
                
                comparisonContent.appendChild(videoItem);
                
                // Add to quadrant chart as historical point
                this.addQuadrantPoint(avgMetrics.attention, avgMetrics.engagement, videoId, video.title, false);
            }
        });
    }
    
    calculateAverageMetrics(metrics) {
        const sum = metrics.reduce((acc, metric) => ({
            attention: acc.attention + (metric.attention || 0),
            engagement: acc.engagement + (metric.engagement || 0),
            workload: acc.workload + (metric.workload || 0),
            alpha_theta: acc.alpha_theta + (metric.alpha_theta || 0)
        }), { attention: 0, engagement: 0, workload: 0, alpha_theta: 0 });
        
        const count = metrics.length;
        return {
            attention: sum.attention / count,
            engagement: sum.engagement / count,
            workload: sum.workload / count,
            alpha_theta: sum.alpha_theta / count
        };
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new NeuroLMDashboard();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, pause updates
    } else {
        // Page is visible, resume updates
        if (window.dashboard && !window.dashboard.isConnected) {
            window.dashboard.connectWebSocket();
        }
    }
});
