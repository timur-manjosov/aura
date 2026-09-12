import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Aura",
  description: "Aura knows your server.",
};

/**
 * The document shell.
 *
 * `lang` is left at the default here rather than being set from the user's
 * locale: the locale is resolved in the browser from `navigator.language`
 * (see app/page.tsx), and a server-rendered `lang` would disagree with the
 * text on the first paint. A later per-user setting, stored server-side,
 * is where a correct `lang` belongs.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
