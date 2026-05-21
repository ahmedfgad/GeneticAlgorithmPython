// Scroll the Furo left sidebar so the active page is centered in view.
// Furo keeps the sidebar at the top by default, so on a tall sidebar the
// current page can be off-screen. This adjusts only the sidebar's own
// scroll position (not the main window).
window.addEventListener("DOMContentLoaded", function () {
  var box = document.querySelector(".sidebar-scroll");
  var active = document.querySelector(".sidebar-tree .current-page");
  if (!box || !active) {
    return;
  }
  var target = active.querySelector("a") || active;
  var delta = target.getBoundingClientRect().top - box.getBoundingClientRect().top;
  box.scrollTop += delta - box.clientHeight / 2 + target.clientHeight / 2;
});
