/* ===================================================
   script.js — House Price Predictor
   =================================================== */

"use strict";

// ─── Config ────────────────────────────────────────
const API_BASE = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") 
    ? (window.location.port === "8000" ? "" : "http://127.0.0.1:8000")
    : "";

// ─── DOM refs ──────────────────────────────────────
const form          = document.getElementById("house-form");
const predictBtn    = document.getElementById("predict-btn");
const loadingOverlay= document.getElementById("loading-overlay");
const resultPanel   = document.getElementById("result-panel");
const resultPrice   = document.getElementById("result-price");
const formError     = document.getElementById("form-error");
const resetBtn      = document.getElementById("reset-btn");
const themeToggle   = document.getElementById("theme-toggle");
const themeIcon     = themeToggle.querySelector(".theme-icon");

// ─── Theme toggle ──────────────────────────────────
(function initTheme() {
  const stored = localStorage.getItem("theme") || "dark";
  document.documentElement.setAttribute("data-theme", stored);
  themeIcon.textContent = stored === "dark" ? "☀️" : "🌙";
})();

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  themeIcon.textContent = next === "dark" ? "☀️" : "🌙";
  localStorage.setItem("theme", next);
});

// ─── Health check ──────────────────────────────────
(async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      if (!data.model_loaded) {
        showError("⚠️ Model is initializing. Predictions may be unavailable for a moment.");
      }
    }
  } catch (err) {
    console.error("Health check failed:", err);
  }
})();

// ─── Sync sliders ──────────────────────────────────
document.querySelectorAll(".slider").forEach((slider) => {
  const valEl = document.getElementById(`${slider.id}-val`);
  if (!valEl) return;
  valEl.textContent = slider.value;
  slider.addEventListener("input", () => { valEl.textContent = slider.value; });
});

// ─── Pill selectors ────────────────────────────────
document.querySelectorAll(".pill-selector").forEach((group) => {
  group.querySelectorAll(".pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      group.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      const hiddenId = pill.dataset.name;
      const hidden = document.getElementById(hiddenId);
      if (hidden) hidden.value = pill.dataset.val;
    });
  });
});

// ─── Error helper ──────────────────────────────────
function showError(msg) {
  formError.textContent = msg;
  formError.classList.add("show");
  formError.style.display = "block";
  setTimeout(() => formError.classList.remove("show"), 4000);
}
function clearError() {
  formError.style.display = "none";
  formError.textContent = "";
}

// ─── Read form values ──────────────────────────────
function readForm() {
  const get = (id) => {
    const el = document.getElementById(id);
    if (!el) return null;
    return el.type === "hidden" || el.tagName === "SELECT" ? el.value : el.value;
  };

  const intFields = ["OverallQual","YearBuilt","HouseAge","GarageCars",
                     "KitchenAbvGr","Fireplaces","FullBath","TotRmsAbvGrd"];
  const floatFields = ["GrLivArea","TotalSF","LotArea"];
  const strFields = ["Neighborhood"];

  const payload = {};
  for (const f of intFields) {
    const v = parseInt(get(f), 10);
    if (isNaN(v)) return { error: `Please enter a valid number for "${f}".` };
    payload[f] = v;
  }
  for (const f of floatFields) {
    const v = parseFloat(get(f));
    if (isNaN(v)) return { error: `Please enter a valid number for "${f}".` };
    payload[f] = v;
  }
  for (const f of strFields) {
    payload[f] = get(f);
  }
  return payload;
}

// ─── Price counter animation ───────────────────────
function animatePrice(target, duration = 1200) {
  const start = performance.now();
  const fmt = new Intl.NumberFormat("en-US", {
    style: "currency", currency: "USD", maximumFractionDigits: 0,
  });

  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = Math.round(eased * target);
    resultPrice.textContent = fmt.format(value);
    if (progress < 1) requestAnimationFrame(step);
    else resultPrice.textContent = fmt.format(target);
  }
  requestAnimationFrame(step);
}

// ─── Confetti ──────────────────────────────────────
function launchConfetti() {
  const container = document.getElementById("confetti-container");
  const colors = ["#6c63ff","#38bdf8","#f472b6","#34d399","#fbbf24","#a78bfa"];
  for (let i = 0; i < 80; i++) {
    const piece = document.createElement("div");
    piece.className = "confetti-piece";
    piece.style.cssText = `
      left: ${Math.random() * 100}%;
      top: -10px;
      background: ${colors[Math.floor(Math.random() * colors.length)]};
      width: ${6 + Math.random() * 8}px;
      height: ${6 + Math.random() * 8}px;
      border-radius: ${Math.random() > 0.5 ? "50%" : "2px"};
      animation-delay: ${Math.random() * 0.8}s;
      animation-duration: ${2 + Math.random() * 1.5}s;
    `;
    container.appendChild(piece);
  }
  setTimeout(() => { container.innerHTML = ""; }, 4000);
}

// ─── Show result ───────────────────────────────────
function showResult(price) {
  form.hidden = true;
  resultPanel.hidden = false;
  resultPrice.textContent = "$0";
  animatePrice(price);
  launchConfetti();
}

// ─── Reset ─────────────────────────────────────────
resetBtn.addEventListener("click", () => {
  resultPanel.hidden = true;
  form.hidden = false;
  clearError();
});

// ─── Submit ────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();

  const payload = readForm();
  if (payload.error) { showError(payload.error); return; }

  // UI: loading state
  predictBtn.disabled = true;
  loadingOverlay.hidden = false;

  try {
    console.log("🚀 Sending prediction request to:", `${API_BASE}/api/predict`);
    
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      if (res.status === 503) {
        showError("⏳ Cold starting... retrying in 3s...");
        setTimeout(() => form.dispatchEvent(new Event("submit")), 3000);
        return;
      }
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    const price = data.predicted_price;
    if (!price || isNaN(price)) throw new Error("Invalid response from server.");

    loadingOverlay.hidden = true;
    predictBtn.disabled = false;
    showResult(price);

  } catch (err) {
    loadingOverlay.hidden = true;
    predictBtn.disabled = false;
    showError(`❌ ${err.message}`);
  }
});

// ─── Smooth scroll for nav links ──────────────────
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const target = document.querySelector(link.getAttribute("href"));
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});

// ─── Intersection observer for feature bars ───────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.animationPlayState = "running";
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.3 });

document.querySelectorAll(".feat-bar").forEach(bar => {
  bar.style.animationPlayState = "paused";
  observer.observe(bar);
});

// ─── Custom Neighborhood Dropdown ──────────────────
(function initCustomDropdown() {
  const dropdown      = document.getElementById('neighborhood-dropdown');
  const trigger       = document.getElementById('neighborhood-trigger');
  const display       = document.getElementById('neighborhood-display');
  const list          = document.getElementById('neighborhood-list');
  const hiddenSelect  = document.getElementById('Neighborhood');

  if (!dropdown || !trigger || !display || !list || !hiddenSelect) return;

  // Toggle dropdown
  trigger.addEventListener('click', () => {
    dropdown.classList.toggle('active');
  });

  // Close when clicking outside
  document.addEventListener('click', (e) => {
    if (!dropdown.contains(e.target)) {
      dropdown.classList.remove('active');
    }
  });

  // Handle selection (no active class, no highlight)
  list.querySelectorAll('li').forEach(item => {
    item.addEventListener('click', () => {
      // Update display text
      display.textContent = item.textContent;
      // Sync hidden select
      hiddenSelect.value = item.dataset.value;
      // Close dropdown
      dropdown.classList.remove('active');
    });
  });

  // Preselect NAmes (only set display, no active class)
  const defaultOption = list.querySelector('li[data-value="NAmes"]');
  if (defaultOption) {
    display.textContent = defaultOption.textContent;
    hiddenSelect.value = defaultOption.dataset.value;
  }
})();

// ─── Shadow Animation (Dark Mode Only) ───────────────────────────
(function() {
  const cards = document.querySelectorAll('.rgb-side-card');
  let intervals = [];

  function startAnimation() {
    // Avoid double-starting
    if (intervals.length > 0) return;
    
    cards.forEach((card, i) => {
      let hue = 0;
      intervals[i] = setInterval(() => {
        hue = (hue + 2) % 360;
        card.style.boxShadow = `0 0 30px hsl(${hue}, 100%, 60%)`;
      }, 50);
    });
  }

  function stopAnimation() {
    intervals.forEach(clearInterval);
    intervals = [];
    cards.forEach(card => {
      card.style.boxShadow = '';   // revert to CSS default
    });
  }

  // Initial check
  if (document.documentElement.getAttribute('data-theme') !== 'light') {
    startAnimation();
  }

  // Listen for theme changes on <html>
  const observer = new MutationObserver(() => {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    if (isLight) {
      stopAnimation();
    } else {
      startAnimation();
    }
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
