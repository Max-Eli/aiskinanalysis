import { NextRequest, NextResponse } from "next/server";

const ROOT_DOMAIN = process.env.NEXT_PUBLIC_ROOT_DOMAIN ?? "localhost";

export function proxy(request: NextRequest) {
  const hostname = request.headers.get("host") ?? "";

  // Strip port for local dev
  const hostWithoutPort = hostname.split(":")[0];

  let tenant = "default";

  if (
    hostWithoutPort !== ROOT_DOMAIN &&
    hostWithoutPort !== `www.${ROOT_DOMAIN}` &&
    hostWithoutPort !== "localhost"
  ) {
    // Extract subdomain: glowspa.yourdomain.com → glowspa
    const parts = hostWithoutPort.split(".");
    if (parts.length >= 2) {
      tenant = parts[0];
    }
  }

  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-tenant", tenant);

  return NextResponse.next({ request: { headers: requestHeaders } });
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
