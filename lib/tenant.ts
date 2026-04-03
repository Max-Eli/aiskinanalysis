import { headers } from "next/headers";
import path from "path";
import fs from "fs";

export interface TenantConfig {
  name: string;
  slug: string;
  logo: string | null;
  primaryColor: string;
  accentColor: string;
  bookingUrl: string | null;
  services: string[];
}

const TENANT_DIR = path.join(process.cwd(), "tenants");

export function getTenantConfig(slug: string): TenantConfig {
  const safeName = slug.replace(/[^a-z0-9-]/gi, "");
  const filePath = path.join(TENANT_DIR, `${safeName}.json`);

  if (fs.existsSync(filePath)) {
    return JSON.parse(fs.readFileSync(filePath, "utf-8")) as TenantConfig;
  }

  // Fall back to default
  const defaultPath = path.join(TENANT_DIR, "default.json");
  return JSON.parse(fs.readFileSync(defaultPath, "utf-8")) as TenantConfig;
}

export async function getCurrentTenant(): Promise<TenantConfig> {
  const headersList = await headers();
  const slug = headersList.get("x-tenant") ?? "default";
  return getTenantConfig(slug);
}
