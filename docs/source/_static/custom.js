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
