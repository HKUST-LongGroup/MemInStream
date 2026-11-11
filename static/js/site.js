(() => {
  'use strict';

  // The initial HTML already contains the abstract, figures, and category results.
  // JavaScript adds navigation, figure previews, and optional result exploration.
  const taskData = JSON.parse(document.getElementById('task-data').textContent);
  const controls = document.getElementById('results-controls');
  const viewSelect = document.getElementById('results-view');
  const filterButtons = [...document.querySelectorAll('[data-filter]')];
  const table = document.querySelector('.results-table');
  const legend = document.getElementById('task-legend');
  const status = document.getElementById('results-status');
  let filter = 'all';

  function updateResults() {
    const view = viewSelect.value;
    let visibleRows = 0;
    table.querySelectorAll('tbody').forEach(body => {
      const group = body.dataset.group;
      const matches = filter === 'all' || filter === group || (filter === 'online' && ['training-free', 'training-based'].includes(group));
      body.hidden = !matches;
      if (matches) visibleRows += body.querySelectorAll('.model-row').length;
    });
    table.querySelectorAll('[data-category]').forEach(cell => {
      cell.hidden = view !== 'all' && cell.dataset.category !== view;
    });
    const columnCount = [...table.querySelectorAll('thead th')].filter(cell => !cell.hidden).length;
    table.querySelectorAll('.group-row th').forEach(cell => { cell.colSpan = columnCount; });
    filterButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.filter === filter)));
    status.textContent = `${visibleRows} configurations · ${viewSelect.options[viewSelect.selectedIndex].text}`;
    legend.replaceChildren();
    const visibleCategories = taskData.categories.filter(([key]) => view === 'all' || view === key);
    visibleCategories.forEach(([, , tasks]) => tasks.forEach(task => {
      const item = document.createElement('span');
      const abbreviation = document.createElement('strong');
      abbreviation.textContent = `${task}:`;
      item.append(abbreviation, document.createTextNode(taskData.tasks[task]));
      legend.append(item);
    }));
    legend.hidden = view === 'averages';
  }

  filterButtons.forEach(button => button.addEventListener('click', () => {
    filter = button.dataset.filter;
    updateResults();
  }));
  viewSelect.addEventListener('change', () => {
    updateResults();
    document.querySelector('.table-scroll').scrollLeft = 0;
  });
  controls.hidden = false;
  updateResults();

  const dialog = document.getElementById('figure-dialog');
  const preview = document.getElementById('dialog-image');
  const original = document.getElementById('original-image');
  let scrollStyle = '';
  if (typeof dialog.showModal === 'function') {
    document.querySelectorAll('[data-zoom]').forEach(link => {
      link.addEventListener('click', event => {
        if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        preview.src = link.href;
        preview.alt = link.querySelector('img').alt;
        original.href = link.href;
        scrollStyle = document.documentElement.style.overflow;
        document.documentElement.style.overflow = 'hidden';
        dialog.showModal();
      });
    });
    document.getElementById('close-dialog').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => {
      if (event.target !== dialog) return;
      const bounds = dialog.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
    });
    dialog.addEventListener('close', () => {
      document.documentElement.style.overflow = scrollStyle;
    });
  }

  const navLinks = [...document.querySelectorAll('.nav-links a')];
  const sections = navLinks.map(link => document.querySelector(link.getAttribute('href')));
  let scheduled = false;
  function updateNavigation() {
    let active = null;
    const threshold = document.querySelector('.site-header').offsetHeight + 100;
    sections.forEach(section => {
      if (section.getBoundingClientRect().top <= threshold) active = section.id;
    });
    navLinks.forEach(link => {
      if (link.getAttribute('href') === `#${active}`) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
    scheduled = false;
  }
  window.addEventListener('scroll', () => {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(updateNavigation);
    }
  }, { passive: true });
  window.addEventListener('resize', updateNavigation);
  updateNavigation();
})();
