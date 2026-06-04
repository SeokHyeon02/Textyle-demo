import { supabase } from '../supabase';

// 찜 목록에서 보여줄 상품 정보 (clothes 테이블에서 join으로 가져옴)
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

// user_bookmarks 한 행 + 연결된 상품 정보
export type BookmarkRow = {
  cloth_id: number;
  created_at: string;
  clothes: BookmarkClothes | null;
};

const CLOTHES_COLUMNS = 'id, image_url, name, brand_name, price, main_category, sub_category, shop_link';

// 현재 유저가 찜한 상품 id 목록 (검색 결과의 하트 상태 초기화용)
export async function fetchBookmarkedIds(userId: string): Promise<number[]> {
  const { data, error } = await supabase
    .from('user_bookmarks')
    .select('cloth_id')
    .eq('user_id', userId);

  if (error) throw error;
  return (data ?? []).map((row) => row.cloth_id as number);
}

// 찜 탭에서 보여줄, 상품 정보까지 포함한 찜 목록.
// user_bookmarks.cloth_id -> clothes.id FK를 통해 PostgREST 임베드 조인으로 한 번에 가져온다.
export async function fetchBookmarks(userId: string): Promise<BookmarkRow[]> {
  const { data, error } = await supabase
    .from('user_bookmarks')
    .select(`cloth_id, created_at, clothes ( ${CLOTHES_COLUMNS} )`)
    .eq('user_id', userId)
    .order('created_at', { ascending: false });

  if (error) throw error;
  return (data ?? []) as unknown as BookmarkRow[];
}

export async function addBookmark(userId: string, clothId: number): Promise<void> {
  const { error } = await supabase
    .from('user_bookmarks')
    .insert({ user_id: userId, cloth_id: clothId });

  if (error) throw error;
}

export async function removeBookmark(userId: string, clothId: number): Promise<void> {
  const { error } = await supabase
    .from('user_bookmarks')
    .delete()
    .eq('user_id', userId)
    .eq('cloth_id', clothId);

  if (error) throw error;
}
