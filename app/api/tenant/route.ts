import { NextResponse } from "next/server";
import { getCurrentTenant } from "@/lib/tenant";

export async function GET() {
  const tenant = await getCurrentTenant();
  return NextResponse.json(tenant);
}
