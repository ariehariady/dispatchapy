document.addEventListener('DOMContentLoaded', function() {
    // Create the main footer element
    const footer = document.createElement('footer');
    footer.className = 'text-center mt-8 text-sm text-gray-500';

    // Create the paragraph element for the watermark text
    const watermarkText = document.createElement('p');
    
    // UPDATED: Use innerHTML to include a clickable mailto link
    watermarkText.innerHTML = 'Copyright (c) 2025 <a href="mailto:ariehariady@gmail.com" class="text-blue-600 hover:underline">ariehariady</a>.';

    // Append the text to the footer, and the footer to the page body
    footer.appendChild(watermarkText);
    document.body.appendChild(footer);
});

