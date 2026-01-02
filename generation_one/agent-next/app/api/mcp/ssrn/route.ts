import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const paperId = searchParams.get('paperId');

  const headers = {
    'Accept': '*/*',
    'Host': 'api.ssrn.com',
    'Connection': 'keep-alive',
    'Cookie': 'SITEID=en; __cf_bm=PbtT79EJ0JrFBGcM9MachqAlLXXx5s2lFzYLngcRljw-1746290434-1.0.1.1-YKlkE0Agkw8A7vgSOamG9tsyEMwPdrclIvxr.sPkejG42VxZqYQ6sCo9T6FyTplJDGvb4NzXhHYaSspuoYdBFfq4BDh7.BmfBsgW4cX94C0; cfid=e754d63b-9e7e-4a4b-99ab-4b06ea8800ff; cftoken=0; CFID=e754d63b-9e7e-4a4b-99ab-4b06ea8800ff; CFTOKEN=0; AWSELB=F583A35D06DDD1DDD942D6006B93D30F25C2908AFACA05CB7D4A12D7071BA57E8A84217F5AD424FB1681E714DD6BE9914D9A96C84C161FD3249A57C94243B9CAF3088409436E833A0E0315B0EF9BD53B1D64C5005D'
  };

  try {
    if (paperId) {
      // Fetch individual paper details
      const response = await fetch(`https://api.ssrn.com/content/v1/papers/${paperId}`, {
        headers
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return NextResponse.json(data);
    } else {
      // Fetch list of papers using the exact URL
      const response = await fetch('https://api.ssrn.com/content/v1/bindings/2978227/papers?index=0&count=50&sort=0', {
        headers
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return NextResponse.json(data);
    }
  } catch (error) {
    console.error('Error fetching SSRN data:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
} 