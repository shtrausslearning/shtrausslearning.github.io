function updateYear() {
    var yearElement = document.getElementById('currentYear');
    if (yearElement) {
        yearElement.textContent = new Date().getFullYear();
    }
}

// Update the year when the site is initially loaded
window.addEventListener('DOMContentLoaded', updateYear);

// Update the year when a new page is loaded
var observer = new MutationObserver(updateYear);
observer.observe(document.body, {childList: true});