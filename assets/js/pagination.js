// Client-side pagination for any [data-paginate="N"] container.
// Items inside the container marked with [data-paginate-item] get sliced
// into pages of N. Pagination controls render at the end of the container.

(function () {
  function init(container) {
    var perPage = parseInt(container.getAttribute('data-paginate'), 10) || 5;
    var items = Array.prototype.slice.call(
      container.querySelectorAll('[data-paginate-item]')
    );
    if (items.length <= perPage) return;

    var totalPages = Math.ceil(items.length / perPage);
    var currentItemIndex = -1;
    for (var i = 0; i < items.length; i++) {
      if (items[i].hasAttribute('data-paginate-current')) {
        currentItemIndex = i;
        break;
      }
    }
    var current = currentItemIndex >= 0
      ? Math.floor(currentItemIndex / perPage) + 1
      : 1;

    var nav = document.createElement('nav');
    nav.className = 'pagination';
    container.appendChild(nav);

    function render() {
      var lastVisible = null;
      items.forEach(function (item, i) {
        var page = Math.floor(i / perPage) + 1;
        var visible = page === current;
        item.style.display = visible ? '' : 'none';
        item.removeAttribute('data-last-visible');
        if (visible) lastVisible = item;
      });
      if (lastVisible) lastVisible.setAttribute('data-last-visible', '');

      while (nav.firstChild) nav.removeChild(nav.firstChild);

      var prev = document.createElement('button');
      prev.type = 'button';
      prev.className = 'pagination__btn';
      prev.textContent = '← prev';
      prev.disabled = current === 1;
      prev.addEventListener('click', function () {
        if (current > 1) { current -= 1; render(); }
      });

      var info = document.createElement('span');
      info.className = 'pagination__info';
      info.textContent = 'page ' + current + ' of ' + totalPages;

      var next = document.createElement('button');
      next.type = 'button';
      next.className = 'pagination__btn';
      next.textContent = 'next →';
      next.disabled = current === totalPages;
      next.addEventListener('click', function () {
        if (current < totalPages) { current += 1; render(); }
      });

      nav.appendChild(prev);
      nav.appendChild(info);
      nav.appendChild(next);
    }

    render();
  }

  function bootstrap() {
    var containers = document.querySelectorAll('[data-paginate]');
    Array.prototype.forEach.call(containers, init);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
  } else {
    bootstrap();
  }
})();

// On mobile post pages, scroll past the stacked sidebar to the post title
// on first navigation so the user lands on the content, not the profile card.
(function () {
  if (window.innerWidth > 880) return;
  if (window.location.hash) return;
  var title = document.querySelector('.post__title');
  if (!title) return;
  var nav = performance.getEntriesByType && performance.getEntriesByType('navigation')[0];
  if (nav && nav.type === 'back_forward') return;
  title.scrollIntoView();
})();
