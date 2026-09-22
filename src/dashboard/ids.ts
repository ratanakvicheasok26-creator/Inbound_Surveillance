export function portraitDataUri(name: string): string {
  const initials = name
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
  const hash = [...name].reduce((sum, ch) => sum + ch.charCodeAt(0), 0);
  const h = hash % 360;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">
    <rect width="96" height="96" fill="hsl(${h} 14% 14%)"/>
    <circle cx="48" cy="38" r="18" fill="hsl(${h} 22% 38%)"/>
    <ellipse cx="48" cy="92" rx="32" ry="28" fill="hsl(${h} 22% 38%)"/>
    <text x="48" y="90" text-anchor="middle" fill="#f4f4f5" font-size="13" font-family="Plus Jakarta Sans, sans-serif" font-weight="600">${initials}</text>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export function uid(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 8)}`;
}

export function padSeq(n: number, width = 4): string {
  return String(n).padStart(width, "0");
}
