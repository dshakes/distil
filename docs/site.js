/* Distil docs — progressive enhancements: copy buttons + "on this page" TOC.
   Vanilla, dependency-free, no external requests. */

/* ── Mobile sidebar toggle: shared by every docs page's ☰ button
   (onclick="toggleSidebar()") ─────────────────────────────────────────
   Keeps #sidebar's "open" class and .sidebar-toggle's aria-expanded in
   sync, closes on outside click or Escape, and returns focus to the
   toggle button on Escape. */
(function () {
  "use strict";

  function getSidebar() { return document.getElementById("sidebar"); }
  function getToggle() { return document.querySelector(".sidebar-toggle"); }

  function setOpen(open) {
    var sb = getSidebar();
    if (!sb) return;
    sb.classList.toggle("open", open);
    var btn = getToggle();
    if (btn) btn.setAttribute("aria-expanded", open ? "true" : "false");
  }

  window.toggleSidebar = function () {
    var sb = getSidebar();
    if (!sb) return;
    setOpen(!sb.classList.contains("open"));
  };

  document.addEventListener("click", function (e) {
    var sb = getSidebar();
    if (sb && sb.classList.contains("open") && !sb.contains(e.target) && !e.target.closest(".sidebar-toggle")) {
      setOpen(false);
    }
  });

  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    var sb = getSidebar();
    if (sb && sb.classList.contains("open")) {
      setOpen(false);
      var btn = getToggle();
      if (btn) btn.focus();
    }
  });
})();

(function () {
  "use strict";

  // ── Copy-to-clipboard on every code block ──────────────────────────
  // Shared visually-hidden polite live region: announces the copy result to
  // screen reader users, since the visual button-label swap alone is not
  // reliably announced.
  var copyStatus = document.createElement("span");
  copyStatus.setAttribute("aria-live", "polite");
  copyStatus.setAttribute("role", "status");
  copyStatus.style.cssText = "position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0";
  document.body.appendChild(copyStatus);
  function announceCopy(msg) {
    copyStatus.textContent = "";
    setTimeout(function () { copyStatus.textContent = msg; }, 50);
  }

  document.querySelectorAll("pre").forEach(function (pre) {
    var target = pre.querySelector("code") || pre;
    var clone = target.cloneNode(true); // strip any pre-existing .copy-btn before reading text
    clone.querySelectorAll(".copy-btn").forEach(function (b) { b.remove(); });
    var original = clone.textContent;
    var btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.type = "button";
    btn.textContent = "Copy";
    btn.addEventListener("click", function () {
      var done = function () {
        btn.textContent = "✓ Copied";
        btn.classList.add("copied");
        announceCopy("Copied to clipboard");
        setTimeout(function () {
          btn.textContent = "Copy";
          btn.classList.remove("copied");
        }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(original).then(done).catch(fallback);
      } else {
        fallback();
      }
      function fallback() {
        var ta = document.createElement("textarea");
        ta.value = original;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try {
          document.execCommand("copy");
          done();
        } catch (e) {
          btn.textContent = "Ctrl-C";
          announceCopy("Copy failed, press Ctrl+C to copy manually");
        }
        document.body.removeChild(ta);
      }
    });
    pre.appendChild(btn);
  });

  // ── "On this page" right-rail TOC built from the content headings ───
  var content = document.querySelector(".content");
  if (!content) return;
  var heads = content.querySelectorAll("h2, h3");
  if (heads.length < 2) return;

  function slug(t) {
    return t.toLowerCase().trim().replace(/[^\w]+/g, "-").replace(/^-+|-+$/g, "");
  }

  var nav = document.createElement("nav");
  nav.className = "toc";
  nav.setAttribute("aria-label", "On this page");
  nav.innerHTML = '<div class="toc-title">On this page</div>';

  var entries = [];
  var seenIds = {};
  var lastH2 = "";
  heads.forEach(function (h) {
    if (!h.id) {
      var base = slug(h.textContent) || "section";
      // Prefix h3 ids with the nearest h2's id so repeated subheadings
      // (e.g. every command's "Flags" section) still get unique, readable ids.
      h.id = (h.tagName === "H3" && lastH2) ? lastH2 + "-" + base : base;
    }
    var n = seenIds[h.id] = (seenIds[h.id] || 0) + 1;
    if (n > 1) h.id = h.id + "-" + n;
    if (h.tagName === "H2") lastH2 = h.id;
    // Clickable "#" ref link on the header itself (deep-link any section).
    if (!h.querySelector(".hanchor")) {
      var ha = document.createElement("a");
      ha.className = "hanchor";
      ha.href = "#" + h.id;
      ha.textContent = "#";
      ha.setAttribute("aria-label", "Link to section: " + h.textContent.trim());
      h.appendChild(ha);
    }
    var a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent.replace(/#/g, "").trim();
    if (h.tagName === "H3") a.className = "lvl-3";
    nav.appendChild(a);
    entries.push({ a: a, h: h });
  });
  content.insertBefore(nav, content.firstChild);

  // ── Scrollspy: highlight the section currently in view ──────────────
  var byId = {};
  entries.forEach(function (e) { byId[e.h.id] = e.a; });
  if ("IntersectionObserver" in window) {
    var obs = new IntersectionObserver(function (records) {
      records.forEach(function (rec) {
        if (rec.isIntersecting) {
          entries.forEach(function (e) { e.a.classList.remove("active"); });
          var act = byId[rec.target.id];
          if (act) act.classList.add("active");
        }
      });
    }, { rootMargin: "-80px 0px -68% 0px", threshold: 0 });
    heads.forEach(function (h) { obs.observe(h); });
  }
})();

/* Tab groups: .tabs > .tab[data-tab] switches .tabpanel[data-panel].
   These are plain toggle buttons (not the ARIA tabs/tablist pattern): each
   button reports its own pressed state via aria-pressed, so they stay
   ordinary Tab-and-Enter-operable buttons with no roving tabindex or
   arrow-key contract to maintain. */
(function () {
  document.querySelectorAll(".tabs").forEach(function (grp) {
    var tabs = grp.querySelectorAll(".tab");
    var panels = grp.querySelectorAll(".tabpanel");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        var key = tab.getAttribute("data-tab");
        tabs.forEach(function (t) {
          var active = t === tab;
          t.classList.toggle("is-active", active);
          t.setAttribute("aria-pressed", active ? "true" : "false");
        });
        panels.forEach(function (p) { p.classList.toggle("is-active", p.getAttribute("data-panel") === key); });
      });
    });
  });
})();

/* Wrap content tables in a focusable, labeled scroll region so keyboard
   users can reach the horizontal scroll that .table-scroll gets on
   narrower viewports (see site.css). Desktop rendering is unaffected. */
(function () {
  "use strict";
  var container = document.querySelector("main.content, .content");
  if (!container) return;
  var lastHeading = null;
  container.querySelectorAll("h1, h2, h3, h4, table").forEach(function (el) {
    if (el.tagName !== "TABLE") {
      lastHeading = el;
      return;
    }
    if (el.closest(".table-scroll")) return; // already wrapped
    var label = lastHeading ? lastHeading.textContent.replace(/#/g, "").trim() : "";
    var wrap = document.createElement("div");
    wrap.className = "table-scroll";
    wrap.setAttribute("tabindex", "0");
    wrap.setAttribute("role", "region");
    wrap.setAttribute("aria-label", label || "Scrollable table");
    el.parentNode.insertBefore(wrap, el);
    wrap.appendChild(el);
  });
})();
