/* Disable auto-scroll for consistent behavior across all API items. */
console.log('[indar-docs] Disabling auto-scroll');

/* Remove hash from URL to prevent browser auto-scroll to anchor. */
if (window.location.hash) {
    window.history.replaceState(null, '', window.location.pathname + window.location.search);
}

/* Block all scrollIntoView calls from theme (applies regardless of prior patches). */
if (typeof Element !== 'undefined' && Element.prototype.scrollIntoView) {
    var original = Element.prototype.scrollIntoView;
    Element.prototype.scrollIntoView = function(options) {
        if (this.closest && this.closest('.nftt-sidebar, .nftt-toc, .toc')) {
            console.log('[indar-docs] Blocked scrollIntoView');
            return;
        }
        return original.call(this, options);
    };
    console.log('[indar-docs] scrollIntoView override active');
}

/* Ensure page stays at top on load in case hash removal doesn't work. */
window.addEventListener('load', function() {
    window.scrollTo(0, 0);
});

/* SciPy-style h1 title: show module path prefix small, name below in full size. */
document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('article h1').forEach(function (h1) {
        var textNode = Array.from(h1.childNodes).find(
            function (n) { return n.nodeType === Node.TEXT_NODE && n.textContent.trim(); }
        );
        if (!textNode) return;

        var text = textNode.textContent.trim();
        var lastDot = text.lastIndexOf('.');
        if (lastDot === -1) return;  /* no dot — module index pages, leave as-is */

        var prefix = text.substring(0, lastDot + 1);
        var name   = text.substring(lastDot + 1);

        var prefixEl = document.createElement('small');
        prefixEl.className = 'h1-prefix';
        prefixEl.textContent = prefix;

        var nameEl = document.createElement('span');
        nameEl.className = 'h1-name';
        nameEl.textContent = name;

        textNode.replaceWith(prefixEl, nameEl);
    });
});
