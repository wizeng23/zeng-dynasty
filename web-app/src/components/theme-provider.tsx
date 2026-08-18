"use client";

// Thin wrapper around next-themes so the app can toggle light/dark.
// It must be a Client Component ("use client") because it uses React context
// and reads the user's system/localStorage theme preference in the browser.
import { ThemeProvider as NextThemesProvider } from "next-themes";
import type { ComponentProps } from "react";

export function ThemeProvider({ children, ...props }: ComponentProps<typeof NextThemesProvider>) {
  return <NextThemesProvider {...props}>{children}</NextThemesProvider>;
}
