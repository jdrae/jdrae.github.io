document.addEventListener('DOMContentLoaded', function() {
    // Configuration / Storage
    const STORAGE_KEY = 'selectedCategory';
    const POSTS_PER_PAGE = 5;

    // DOM elements
    const tagEls = document.querySelectorAll('.category .tag-name');
    const titleCategoryEl = document.querySelector('.posts-title .posts-category');
    const titleCountEl = document.querySelector('.posts-title .posts-cnt');
    const postsContainer = document.getElementById('posts-container');
    const paginationContainer = document.querySelector('.pagination');

    // State
    let currentCategory = 'All';
    let currentPage = 1;
    let filteredPosts = [];

    function normalize(str) {
        return (str || '').toString().trim().toLowerCase();
    }

    function saveSelectedCategory(category) {
        try { localStorage.setItem(STORAGE_KEY, category); } catch (e) {}
    }

    function getSelectedCategory() {
        try {
            return localStorage.getItem(STORAGE_KEY) || 'All';
        } catch (e) {
            return 'All';
        }
    }

    function isReloadNavigation() {
        try {
            const nav = performance.getEntriesByType && performance.getEntriesByType('navigation');
            if (nav && nav.length) return nav[0].type === 'reload';
            // Fallback (deprecated API)
            return performance && performance.navigation && performance.navigation.type === 1;
        } catch (e) {
            return false;
        }
    }

    function setActiveCategory(category) {
        tagEls.forEach(function(tag) { tag.classList.remove('active'); });

        const selectedTag = Array.from(tagEls).find(function(tag) {
            const tagCategory = tag.getAttribute('data-category') || tag.textContent;
            return normalize(tagCategory) === normalize(category);
        });

        if (selectedTag) {
            selectedTag.classList.add('active');
        } else if (tagEls[0]) {
            tagEls[0].classList.add('active');
            category = tagEls[0].getAttribute('data-category') || tagEls[0].textContent;
        }

        if (titleCategoryEl) {
            titleCategoryEl.textContent = category;
        }
    }

    function createPostRow(post) {
        const row = document.createElement('tr');
        row.className = 'post-row';
        row.setAttribute('data-categories', post.categories.join(','));
        row.innerHTML = `
            <td class="post-title">
                <a href="${post.url}">${post.title}</a>
            </td>
            <td class="post-date">${post.date}</td>
        `;
        return row;
    }

    function applyFilter(category) {
        currentCategory = category;
        if (category === 'All') {
            filteredPosts = Array.isArray(allPosts) ? allPosts.slice() : [];
        } else {
            filteredPosts = (Array.isArray(allPosts) ? allPosts : []).filter(function(post) {
                return post.categories && post.categories.some(function(cat) {
                    return normalize(cat) === normalize(category);
                });
            });
        }
    }

    function getCurrentPagePosts() {
        const startIndex = (currentPage - 1) * POSTS_PER_PAGE;
        const endIndex = startIndex + POSTS_PER_PAGE;
        return filteredPosts.slice(startIndex, endIndex);
    }

    function updateDisplay() {
        if (titleCountEl) {
            titleCountEl.textContent = filteredPosts.length;
        }

        if (postsContainer) {
            const currentPagePosts = getCurrentPagePosts();
            postsContainer.innerHTML = '';
            for (let i = 0; i < POSTS_PER_PAGE; i++) {
                if (i < currentPagePosts.length) {
                    postsContainer.appendChild(createPostRow(currentPagePosts[i]));
                } else {
                    const emptyRow = document.createElement('tr');
                    emptyRow.className = 'post-row empty-row';
                    emptyRow.innerHTML = `
                        <td class="post-title"></td>
                        <td class="post-date"></td>
                    `;
                    postsContainer.appendChild(emptyRow);
                }
            }
        }

        updatePagination();
    }

    function updatePagination() {
        if (!paginationContainer) return;

        const totalPages = Math.ceil(filteredPosts.length / POSTS_PER_PAGE) || 1;

        let paginationHTML = '';
        if (currentPage > 1) {
            paginationHTML += `<span class="page-btn" data-page="${currentPage - 1}">&lt;</span>`;
        } else {
            paginationHTML += `<span class="page-btn disabled">&lt;</span>`;
        }

        const maxVisiblePages = 5;
        let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
        let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
        if (endPage - startPage + 1 < maxVisiblePages) {
            startPage = Math.max(1, endPage - maxVisiblePages + 1);
        }

        for (let i = startPage; i <= endPage; i++) {
            if (i === currentPage) {
                paginationHTML += `<span class="page-btn active">${i}</span>`;
            } else {
                paginationHTML += `<span class="page-btn" data-page="${i}">${i}</span>`;
            }
        }

        if (currentPage < totalPages) {
            paginationHTML += `<span class="page-btn" data-page="${currentPage + 1}">&gt;</span>`;
        } else {
            paginationHTML += `<span class="page-btn disabled">&gt;</span>`;
        }

        paginationContainer.innerHTML = paginationHTML;

        const pageButtons = paginationContainer.querySelectorAll('.page-btn[data-page]');
        pageButtons.forEach(function(button) {
            button.addEventListener('click', function() {
                const page = parseInt(this.getAttribute('data-page'));
                if (!isNaN(page) && page !== currentPage) {
                    currentPage = page;
                    updateDisplay();
                }
            });
        });
    }

    // Tag click handlers: save selection and re-render
    tagEls.forEach(function(el) {
        el.addEventListener('click', function() {
            tagEls.forEach(function(t) { t.classList.remove('active'); });
            this.classList.add('active');
            const category = this.getAttribute('data-category') || this.textContent;
            saveSelectedCategory(category);
            setActiveCategory(category);
            currentPage = 1;
            applyFilter(category);
            updateDisplay();
        });
    });

    // Initialize selection
    const hasStored = (function() { try { return !!localStorage.getItem(STORAGE_KEY); } catch (e) { return false; } })();
    if (isReloadNavigation() || !hasStored) {
        saveSelectedCategory('All');
    }

    currentCategory = getSelectedCategory();
    setActiveCategory(currentCategory);
    applyFilter(currentCategory);
    updateDisplay();
});