'use client';

export interface WorldQuantCredentials {
  username: string;
  password: string;
  jwtToken?: string;
}

export async function authenticateWorldQuant({ username, password, jwtToken }: WorldQuantCredentials) {
  try {
    // Store credentials for future use
    localStorage.setItem('worldquant_credentials', JSON.stringify({ username, password }));
    
    // Store the JWT token if provided
    if (jwtToken) {
      localStorage.setItem('worldquant_jwt', jwtToken);
    }
    
    // Test the credentials by making a request to our authentication API route
    const response = await fetch('/api/auth', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(jwtToken ? { 'Cookie': `t=${jwtToken}` } : {}),
      },
      body: JSON.stringify({ username, password, jwtToken }),
      credentials: 'include', // Important: This tells fetch to include cookies
    });

    if (!response.ok) {
      throw new Error('Authentication failed');
    }

    const data = await response.json();
    
    // Store the JWT token if available from the response
    if (data.jwtToken) {
      localStorage.setItem('worldquant_jwt', data.jwtToken);
    }

    return true;
  } catch (error) {
    console.error('Authentication error:', error);
    throw error;
  }
}

export function getStoredCredentials(): WorldQuantCredentials | null {
  if (typeof window === 'undefined') {
    return null;
  }
  
  const storedCredentials = localStorage.getItem('worldquant_credentials');
  const storedJWT = localStorage.getItem('worldquant_jwt');
  if (!storedCredentials) {
    return null;
  }
  
  try {
    const credentials = JSON.parse(storedCredentials) as WorldQuantCredentials;
    return { ...credentials, jwtToken: storedJWT || undefined };
  } catch (error) {
    console.error('Error parsing stored credentials:', error);
    return null;
  }
}

export function storeCredentials(credentials: WorldQuantCredentials): void {
  if (typeof window === 'undefined') {
    return;
  }
  
  localStorage.setItem('worldquant_credentials', JSON.stringify(credentials));
}

export function clearStoredCredentials() {
  if (typeof window === 'undefined') return;
  localStorage.removeItem('worldquant_credentials');
}

export async function makeWorldQuantRequest(endpoint: string, options: RequestInit = {}) {
  if (typeof window === 'undefined') {
    throw new Error('This function can only be called in the browser');
  }

  const credentials = getStoredCredentials();

  if (!credentials) {
    throw new Error('Not authenticated');
  }

  // Use the proxy API route for all WorldQuant Brain API requests
  const url = new URL('/api/worldquant', window.location.origin);
  url.searchParams.set('endpoint', endpoint);

  const response = await fetch(url.toString(), {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...options.headers,
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });

  if (response.status === 401) {
    // Authentication failed, clear credentials
    clearStoredCredentials();
    throw new Error('Authentication required');
  }

  return response;
}

export function getStoredCookies(): string | null {
  if (typeof window === 'undefined') {
    return null;
  }
  
  return localStorage.getItem('worldquant_cookies');
}

export function getStoredJWT(): string | null {
  if (typeof window === 'undefined') {
    // Server-side rendering - localStorage not available
    return null;
  }
  
  return localStorage.getItem('worldquant_jwt');
} 