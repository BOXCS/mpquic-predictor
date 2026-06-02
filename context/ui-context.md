# UI Context

## Theme

Light and dark mode are both supported via Tailwind CSS `dark:` variants
and CSS custom properties. The default is system-preference-based.
The design language is a clean technical dashboard — neutral surfaces,
clear data hierarchy, and blue accent colors for interactive elements,
active states, and chart highlights. Avoid decorative elements;
every visual component must serve a monitoring or data purpose.

## Colors

All components must use these CSS custom properties — no hardcoded hex values.
Define these in `web/frontend/src/index.css` under `:root` and `.dark`.

| Role              | CSS Variable          | Light Value | Dark Value  |
| ----------------- | --------------------- | ----------- | ----------- |
| Page background   | `--bg-base`           | `#f9fafb`   | `#0f1117`   |
| Surface           | `--bg-surface`        | `#ffffff`   | `#1a1d27`   |
| Surface raised    | `--bg-surface-raised` | `#f3f4f6`   | `#22263a`   |
| Primary text      | `--text-primary`      | `#111827`   | `#f1f5f9`   |
| Muted text        | `--text-muted`        | `#6b7280`   | `#94a3b8`   |
| Primary accent    | `--accent-primary`    | `#2563eb`   | `#3b82f6`   |
| Accent subtle     | `--accent-subtle`     | `#eff6ff`   | `#1e3a5f`   |
| Border            | `--border-default`    | `#e5e7eb`   | `#2d3148`   |
| Success           | `--state-success`     | `#16a34a`   | `#22c55e`   |
| Warning           | `--state-warning`     | `#d97706`   | `#f59e0b`   |
| Error / Danger    | `--state-error`       | `#dc2626`   | `#ef4444`   |
| Chart path 0      | `--chart-path-0`      | `#2563eb`   | `#3b82f6`   |
| Chart path 1      | `--chart-path-1`      | `#16a34a`   | `#22c55e`   |
| Chart degraded    | `--chart-degraded`    | `#dc2626`   | `#ef4444`   |
| Chart prediction  | `--chart-prediction`  | `#d97706`   | `#f59e0b`   |

> Chart tokens are consumed directly by Recharts `stroke` and `fill` props.
> Never use raw hex values inside chart components.

## Typography

| Role      | Font        | Tailwind Class      | Variable      |
| --------- | ----------- | ------------------- | ------------- |
| UI text   | Inter       | `font-sans`         | `--font-sans` |
| Mono/code | JetBrains Mono | `font-mono`      | `--font-mono` |

Load both via Google Fonts in `index.html`. Use `font-mono` for
metric values (RTT ms, goodput Mbps) to improve readability of
rapidly updating numbers.

## Border Radius

| Context            | Tailwind Class  |
| ------------------ | --------------- |
| Badges / pills     | `rounded-full`  |
| Buttons / inputs   | `rounded-md`    |
| Cards / panels     | `rounded-lg`    |
| Modals / overlays  | `rounded-xl`    |

## Component Library

Tailwind CSS utility classes only — no external component library.
All reusable components are hand-built and live in
`web/frontend/src/components/`. Do not install shadcn/ui or
similar libraries without explicit instruction.

Existing components to build and maintain:
- `RTTChart` — Recharts LineChart showing RTT per path over time
- `PathStatus` — badge showing active path (wlan0 / eth0) and health state
- `AlertBanner` — dismissible banner triggered on switching events or
  predicted degradation; uses `--state-warning` or `--state-error`

## Layout Patterns

- **Dashboard page**: full-viewport layout with a fixed top navbar,
  a two-column grid (chart area left, path status + alert right),
  and a metric summary row at the top showing RTT, goodput, and
  prediction confidence as stat cards
- **History page**: full-width table or timeline of past switching
  events and metric snapshots, paginated from SQLite via REST API
- **Scenario page**: list of the 14 degradation scenarios with
  status indicators showing which have been simulated and logged
- **Navbar**: fixed top bar with project name, active page indicator,
  and a live connection status dot (green = WebSocket connected,
  red = disconnected); no sidebar
- **Cards / panels**: white surface (`--bg-surface`) with a 1px
  border (`--border-default`) and `rounded-lg`; no drop shadows
- **Modals**: centered overlay with a semi-transparent backdrop;
  `rounded-xl` panel, closes on backdrop click or Escape key

## Icons

Lucide React. Stroke-based icons only.

| Context              | Size class    |
| -------------------- | ------------- |
| Inline / label icons | `h-4 w-4`     |
| Button icons         | `h-5 w-5`     |
| Status indicators    | `h-5 w-5`     |
| Empty state / hero   | `h-8 w-8`     |

Recommended icons by use case:
- `Activity` — RTT / network activity chart
- `Wifi`, `Network` — path status (wlan0, eth0)
- `AlertTriangle` — degradation warning
- `ArrowLeftRight` — path switching event
- `CheckCircle` — healthy path state
- `XCircle` — degraded or failed path
- `Clock` — timestamps and history
- `BarChart2` — goodput / throughput metric
