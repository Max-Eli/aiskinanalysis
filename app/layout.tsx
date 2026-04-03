import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";
import { TenantProvider } from "@/components/TenantProvider";
import { getCurrentTenant } from "@/lib/tenant";

const geist = Geist({ subsets: ["latin"], variable: "--font-geist" });

export const metadata: Metadata = {
  title: "Skin Analysis",
  description: "AI-powered skin analysis and personalized treatment recommendations",
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const tenant = await getCurrentTenant();

  return (
    <html lang="en" className={`${geist.variable} h-full`}>
      <body className="min-h-full bg-stone-50 font-sans antialiased">
        <TenantProvider initial={tenant}>{children}</TenantProvider>
      </body>
    </html>
  );
}
