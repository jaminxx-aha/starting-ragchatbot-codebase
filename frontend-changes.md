# Frontend Changes - Theme Toggle Button & Light Theme Variant

## Overview
Added a theme toggle button that allows users to switch between dark and light themes. The button is positioned in the top-right corner of the screen and features smooth transition animations. Also implemented a complete light theme variant with proper colors for accessibility.

## Files Modified

### 1. frontend/style.css

#### Dark Theme (Default) CSS Variables (`:root`)
- **Core colors**:
  - Background: `#0f172a` (dark blue-gray)
  - Surface: `#1e293b` (slightly lighter)
  - Surface hover: `#334155`
  - Text primary: `#f1f5f9` (near white)
  - Text secondary: `#94a3b8` (gray)
  - Border: `#334155`
  - Primary color: `#2563eb` (blue)
  - Primary hover: `#1d4ed8`

- **Message colors**:
  - User message: `#2563eb` (blue background, white text)
  - Assistant message: `#374151` (dark gray)

- **Dark theme specific variables**:
  - Source link background: `rgba(37, 99, 235, 0.15)`
  - Source link border: `rgba(37, 99, 235, 0.3)`
  - Source link hover: `rgba(37, 99, 235, 0.25)`
  - Source text background: `rgba(148, 163, 184, 0.15)`
  - Source text border: `rgba(148, 163, 184, 0.2)`
  - Code/pre background: `rgba(0, 0, 0, 0.2)`
  - Link color: `#60a5fa` (lighter blue for dark backgrounds)

#### Light Theme CSS Variables (`[data-theme="light"]`)
- **Core colors with good contrast**:
  - Background: `#f8fafc` (very light gray, WCAG AA compliant)
  - Surface: `#e2e8f0` (light gray for cards/sidebars)
  - Surface hover: `#cbd5e1`
  - Text primary: `#1e293b` (dark blue-gray, contrast ratio ~12:1 with background)
  - Text secondary: `#64748b` (medium gray)
  - Border: `#cbd5e1`
  - Primary color: `#2563eb` (same blue, works well on light backgrounds)
  - Primary hover: `#1d4ed8`

- **Message colors**:
  - User message: `#2563eb` (blue background, white text for contrast)
  - Assistant message: `#ffffff` (white background, dark text)
  - User message text: `#ffffff` (white, contrast ratio ~4.5:1 with blue background)

- **Light theme specific variables**:
  - Source link background: `rgba(37, 99, 235, 0.1)` (lighter opacity)
  - Source link border: `rgba(37, 99, 235, 0.25)`
  - Source link hover: `rgba(37, 99, 235, 0.2)`
  - Source text background: `rgba(100, 116, 139, 0.1)`
  - Source text border: `rgba(100, 116, 139, 0.2)`
  - Code/pre background: `rgba(0, 0, 0, 0.05)` (subtle gray)
  - Link color: `#2563eb` (primary blue)

- **Theme toggle button styles**:
  - Fixed position in top-right corner (`top: 1rem; right: 1rem`)
  - 44px x 44px button with 12px rounded corners
  - Background and border using CSS variables for theme compatibility
  - Hover effect with lift animation (`translateY(-2px)`)
  - Focus ring for accessibility
  - Icon rotation and opacity transitions for smooth sun/moon switching

- **Transition effects** for smooth theme switching on all key elements

#### Updated styles to use CSS variables:
- `.source-link` and `.source-text` - now use theme-aware variables
- `.message-content code` and `.message-content pre` - use `--code-bg` and `--pre-bg`
- `.message-content blockquote` - fixed to use `--primary-color` instead of undefined `--primary`
- `.message-content a` - added link styles with `--link-color`
- `.message.user .message-content` - uses `--user-message-text` variable
- `.message.welcome-message .message-content` - uses `--shadow` variable

### 2. frontend/index.html
- **Added theme toggle button HTML** at the top of `<body>`:
  - `<button class="theme-toggle" id="themeToggle">` with:
    - `aria-label="Toggle theme"` for accessibility
    - `title="Toggle light/dark theme"` for tooltip
    - Sun icon SVG (visible in dark mode)
    - Moon icon SVG (visible in light mode)

- **Updated CSS/JS version numbers** for cache refresh

### 3. frontend/script.js
- **Added global state**:
  - `currentTheme` - tracks current theme ('dark' or 'light')

- **Added theme functions**:
  - `initTheme()` - loads saved theme from localStorage, defaults to 'dark'
  - `applyTheme(theme)` - applies theme to document and updates aria-label/title
  - `toggleTheme()` - switches between dark and light themes, saves to localStorage

- **Updated `setupEventListeners()`**:
  - Added event listener for `#themeToggle` button click

## Features

### Theme Toggle Button
1. **Icon-based design**: Sun icon (dark mode) and Moon icon (light mode) with rotation animation
2. **Positioned in top-right**: Fixed position that stays visible regardless of scroll
3. **Smooth transitions**: 0.3s ease transitions for all color changes
4. **Accessible and keyboard-navigable**:
   - `<button>` element (native focus and keyboard support)
   - `aria-label` for screen readers (updates dynamically: "切换到浅色主题"/"切换到深色主题")
   - `title` for tooltip
   - Focus ring on focus state
   - Enter and Space keys work by default

### Light Theme Colors (Accessibility)
1. **High contrast text**: 
   - Primary text `#1e293b` on background `#f8fafc` (contrast ratio ~12:1, exceeds WCAG AAA)
   - Secondary text `#64748b` (still readable for less important content)

2. **User message contrast**:
   - Blue background `#2563eb` with white text `#ffffff` (contrast ratio ~4.5:1, meets WCAG AA)

3. **Link visibility**:
   - Primary blue `#2563eb` on light backgrounds (clearly visible)
   - Lighter blue `#60a5fa` on dark backgrounds (good contrast)

4. **Source tags**:
   - Adjusted opacity for visibility on light backgrounds
   - Maintained distinction between clickable links and plain text sources

5. **Code blocks**:
   - Subtle gray background (`rgba(0, 0, 0, 0.05)`) for light theme
   - Darker background (`rgba(0, 0, 0, 0.2)`) for dark theme

## Theme Persistence
- Theme preference is saved in `localStorage` as `'theme'` key
- On page load, saved theme is restored (defaults to 'dark' if not set)