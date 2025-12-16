document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("img.zoomable").forEach(img => {
    img.style.cursor = "zoom-in";

    img.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();

      const overlay = document.createElement("div");
      overlay.className = "img-overlay";

      const full = document.createElement("img");
      full.src = img.src;
      full.alt = img.alt;

      overlay.appendChild(full);
      document.body.appendChild(overlay);

      overlay.addEventListener("click", () => {
        overlay.remove();
      });
    });
  });
});

