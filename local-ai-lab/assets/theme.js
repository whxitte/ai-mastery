// Runs before first paint so the page never flashes the wrong theme.
(function () {
  var t;
  try { t = JSON.parse(localStorage.getItem("lal-theme")); } catch (e) {}
  if (t !== "light" && t !== "dark") t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.dataset.theme = t;
})();
