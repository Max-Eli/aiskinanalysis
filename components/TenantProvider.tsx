"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from "react";

export interface TenantConfig {
  name: string;
  slug: string;
  logo: string | null;
  primaryColor: string;
  accentColor: string;
  bookingUrl: string | null;
  services: string[];
}

const TenantContext = createContext<TenantConfig | null>(null);

export function TenantProvider({
  initial,
  children,
}: {
  initial: TenantConfig;
  children: ReactNode;
}) {
  const [tenant] = useState<TenantConfig>(initial);

  useEffect(() => {
    document.documentElement.style.setProperty(
      "--primary",
      tenant.primaryColor
    );
    document.documentElement.style.setProperty(
      "--accent",
      tenant.accentColor
    );
  }, [tenant]);

  return (
    <TenantContext.Provider value={tenant}>{children}</TenantContext.Provider>
  );
}

export function useTenant(): TenantConfig {
  const ctx = useContext(TenantContext);
  if (!ctx) throw new Error("useTenant must be used inside TenantProvider");
  return ctx;
}
