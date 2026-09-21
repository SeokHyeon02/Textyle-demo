import AsyncStorage from '@react-native-async-storage/async-storage';

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

const getApiUrl = () => process.env.EXPO_PUBLIC_FASHION_API_URL?.replace(/\/$/, '') || 'http://localhost:8080/api';

const getHeaders = async () => {
  const token = await AsyncStorage.getItem('userToken');
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };
};

export async function fetchBookmarkedIds(userId: string): Promise<number[]> {
  const response = await fetch(`${getApiUrl()}/bookmarks`, {
    headers: await getHeaders(),
  });
  if (!response.ok) throw new Error('Failed to fetch bookmarked ids');
  
  const data: BookmarkRow[] = await response.json();
  return data.map((row) => row.cloth_id);
}

export async function fetchBookmarks(userId: string): Promise<BookmarkRow[]> {
  const response = await fetch(`${getApiUrl()}/bookmarks`, {
    headers: await getHeaders(),
  });
  if (!response.ok) throw new Error('Failed to fetch bookmarks');
  
  return await response.json();
}

export async function addBookmark(userId: string, clothId: number): Promise<void> {
  const response = await fetch(`${getApiUrl()}/bookmarks/${clothId}`, {
    method: 'POST',
    headers: await getHeaders(),
  });
  if (!response.ok) throw new Error('Failed to add bookmark');
}

export async function removeBookmark(userId: string, clothId: number): Promise<void> {
  const response = await fetch(`${getApiUrl()}/bookmarks/${clothId}`, {
    method: 'DELETE',
    headers: await getHeaders(),
  });
  if (!response.ok) throw new Error('Failed to remove bookmark');
}
