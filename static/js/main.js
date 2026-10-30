document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('diagnostic-form');
    const fileInput = document.getElementById('file-input');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const emptyState = document.getElementById('empty-state');
    const resultsBody = document.getElementById('results-body');
    const cropSelect = document.getElementById('crop_species');

    // Result elements
    const resCropName = document.getElementById('res-crop-name');
    const resDiseaseName = document.getElementById('res-disease-name');
    const resConfidence = document.getElementById('res-confidence');
    const resSeverityBadge = document.getElementById('res-severity-badge');
    const resSeverityStage = document.getElementById('res-severity-stage');
    const resSeverityConf = document.getElementById('res-severity-conf');
    const severityFill = document.getElementById('severity-fill');
    const resImageDisplay = document.getElementById('res-image-display');
    const statusTag = document.getElementById('status-tag');

    // Mismatch Alert elements
    const mismatchBanner = document.getElementById('mismatch-banner');
    const mismatchText = document.getElementById('mismatch-text');
    const suggestedCropName = document.getElementById('suggested-crop-name');
    const autoFixBtn = document.getElementById('auto-fix-btn');

    // Grad-CAM Heatmap Toggle buttons
    const btnShowHeatmap = document.getElementById('btn-show-heatmap');
    const btnShowOriginal = document.getElementById('btn-show-original');

    let currentOriginalUrl = '';
    let currentHeatmapUrl = '';
    let lastSuggestedSpecies = '';

    // File Preview Handler
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                imagePreview.src = event.target.result;
                dropzonePrompt.classList.add('hidden');
                previewContainer.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropzonePrompt.classList.remove('hidden');
    });

    // Toggle Grad-CAM / Original Image
    btnShowHeatmap.addEventListener('click', () => {
        btnShowHeatmap.classList.add('active');
        btnShowOriginal.classList.remove('active');
        if (currentHeatmapUrl) resImageDisplay.src = currentHeatmapUrl;
    });

    btnShowOriginal.addEventListener('click', () => {
        btnShowOriginal.classList.add('active');
        btnShowHeatmap.classList.remove('active');
        if (currentOriginalUrl) resImageDisplay.src = currentOriginalUrl;
    });

    // Auto-fix Mismatch Click
    autoFixBtn.addEventListener('click', () => {
        if (lastSuggestedSpecies) {
            cropSelect.value = lastSuggestedSpecies;
            form.dispatchEvent(new Event('submit'));
        }
    });

    // Form Submission Handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!fileInput.files[0]) {
            alert('Please select or drop a leaf image file first.');
            return;
        }

        const formData = new FormData(form);
        loadingOverlay.classList.remove('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            loadingOverlay.classList.add('hidden');

            if (!response.ok || !data.success) {
                alert('Error: ' + (data.error || 'Failed to analyze image'));
                return;
            }

            // Display Results
            emptyState.classList.add('hidden');
            resultsBody.classList.remove('hidden');
            statusTag.textContent = 'Analysis Complete';
            statusTag.style.color = '#10b981';

            // Populate Crop Name, Disease Name & Confidence
            resCropName.textContent = data.crop_name;
            resDiseaseName.textContent = data.disease_name;
            resConfidence.textContent = `${data.confidence}%`;

            // Populate Severity
            resSeverityStage.textContent = data.severity_stage;
            resSeverityBadge.className = `severity-badge ${data.severity_stage}`;
            resSeverityConf.textContent = `${data.severity_confidence}%`;

            if (data.severity_stage === 'Early') {
                severityFill.style.width = '25%';
            } else if (data.severity_stage === 'Moderate') {
                severityFill.style.width = '60%';
            } else {
                severityFill.style.width = '100%';
            }

            // Store & Display Images
            currentOriginalUrl = data.original_image_url;
            currentHeatmapUrl = data.heatmap_image_url;
            resImageDisplay.src = currentHeatmapUrl;
            btnShowHeatmap.classList.add('active');
            btnShowOriginal.classList.remove('active');

            // Handle Mismatch Warning (Objective 4)
            if (data.mismatch_alert) {
                mismatchBanner.classList.remove('hidden');
                lastSuggestedSpecies = data.suggested_species;
                suggestedCropName.textContent = data.suggested_species;
                mismatchText.textContent = `You selected '${data.selected_species}', but AI detected leaf features matching '${data.suggested_species}'.`;
            } else {
                mismatchBanner.classList.add('hidden');
            }

        } catch (err) {
            loadingOverlay.classList.add('hidden');
            alert('An unexpected error occurred during prediction: ' + err.message);
        }
    });
});
