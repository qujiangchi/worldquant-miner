import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // Since we're using sessionStorage, we can't check authentication in middleware
  // Instead, we'll let the client-side components handle authentication redirects
  // This middleware is kept for future use if needed
  
  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
}; 