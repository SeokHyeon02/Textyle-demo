import { supabase } from '../supabase';

export type BookmarkClothes = {
  id: number;
  image_url: string | null;
  name: string | null;
  brand_name: string | null;
  price: number | string | null;
  main_category: string | null;
  sub_category: string | null;
  shop_link: string | null;
};

export type BookmarkRow = {
  cloth_id: number;
  created_at: string;
  clothes: BookmarkClothes | null;
};

const getApiUrl = () => process.env.EXPO_PUBLIC_FASHION_API_URL?.replace(/\/$/, '') || 'http://localhost:8080';

const getHeaders = async () => {
  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session) {
    throw new Error('로그인이 필요합니다.');
  }

  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${session.access_token}`,
  };
};

export async function fetchBookmarkedIds(): Promise<number[]> {
  const response = await fetch(`${getApiUrl()}/api/bookmarks`, {
    headers: await getHeaders(),
  });
  if (!response.ok) {
    throw new Error('Failed to fetch bookmarked ids');
  }

  const data: BookmarkRow[] = await response.json();

  return data.map((row) => row.cloth_id);
}


export async function fetchBookmarks(): Promise<BookmarkRow[]> {
  const response = await fetch(`${getApiUrl()}/api/bookmarks`, {
    headers: await getHeaders(),
  });
  if (!response.ok) throw new Error('Failed to fetch bookmarks');
  
  return await response.json();
}

export async function addBookmark(clothId: number): Promise<void> {
  const response = await fetch(`${getApiUrl()}/api/bookmarks/${clothId}`, {
    method: 'POST',
    headers: await getHeaders(),
  });
  if (!response.ok) throw new Error('Failed to add bookmark');
}

export async function removeBookmark(clothId: number): Promise<void> {
  const response = await fetch(`${getApiUrl()}/api/bookmarks/${clothId}`, {
    method: 'DELETE',
    headers: await getHeaders(),
  });
  if (!response.ok) throw new Error('Failed to remove bookmark');
}
