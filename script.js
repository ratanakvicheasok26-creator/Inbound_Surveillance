(function () {
  const menu = document.getElementById("mobile-menu");
  const toggle = document.querySelector("[data-menu-toggle]");
  const form = document.getElementById("contact-form");
  const done = document.getElementById("form-done");
  const links = document.querySelectorAll(".glass-nav a, .mobile-menu nav a");
  const sections = ["home", "about", "solutions", "how", "pricing", "contact"]
    .map((id) => document.getElementById(id))
    .filter(Boolean);

  if (toggle && menu) {
    toggle.addEventListener("click", () => {
      const open = menu.hidden;
      menu.hidden = !open;
      toggle.setAttribute("aria-expanded", String(open));
      toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    });

    menu.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        menu.hidden = true;
        toggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  if (form && done) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      form.querySelectorAll("label, button[type='submit']").forEach((el) => {
        el.hidden = true;
      });
      done.hidden = false;
    });
  }

  const setActive = (id) => {
    links.forEach((link) => {
      const match = link.getAttribute("href") === `#${id}`;
      link.classList.toggle("is-active", match);
    });
  };

  if ("IntersectionObserver" in window && sections.length) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) setActive(entry.target.id);
        });
      },
      { rootMargin: "-40% 0px -50% 0px", threshold: 0 }
    );
    sections.forEach((section) => io.observe(section));
  }
})();
