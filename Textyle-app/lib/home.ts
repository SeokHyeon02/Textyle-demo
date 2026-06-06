import { supabase } from '../supabase';

export type Product = {
  id: number;
  image_url: string | null;
  brand_name: string | null;
  name: string | null;
  price: number | string | null;
  sub_category: string | null;
  shop_link: string | null;
};

export const HOME_PAGE_SIZE = 20;

const PRODUCT_COLUMNS = 'id, image_url, brand_name, name, price, sub_category, shop_link';

let categoriesCache: string[] | null = null;

// 카테고리(sub_category) 목록.
// 빠른 경로는 RPC(get_sub_categories) 이고, 아직 만들지 않았으면 클라이언트 스캔으로 폴백한다.
export async function fetchCategories(): Promise<string[]> {
  if (categoriesCache) return categoriesCache;

  const { data, error } = await supabase.rpc('get_sub_categories');
  if (!error && Array.isArray(data)) {
    categoriesCache = (data as { sub_category: string | null }[])
      .map((row) => row.sub_category)
      .filter((value): value is string => !!value);
    return categoriesCache;
  }

  categoriesCache = await scanDistinctCategories();
  return categoriesCache;
}

// RPC가 없을 때: clothes 의 sub_category 컬럼만 페이지로 훑어 중복 제거.
async function scanDistinctCategories(): Promise<string[]> {
  const seen = new Set<string>();
  const step = 1000;
  let from = 0;

  for (let i = 0; i < 20; i++) {
    const { data, error } = await supabase
      .from('clothes')
      .select('sub_category')
      .range(from, from + step - 1);

    if (error) throw error;
    const rows = (data ?? []) as { sub_category: string | null }[];
    for (const row of rows) {
      if (row.sub_category) seen.add(row.sub_category);
    }
    if (rows.length < step) break;
    from += step;
  }

  return Array.from(seen);
}

// 카테고리별 상품을 페이지 단위로 가져온다. category 가 null 이면 전체.
// 반환 개수가 HOME_PAGE_SIZE 보다 작으면 더 이상 없음.
export async function fetchProducts(category: string | null, page: number): Promise<Product[]> {
  let query = supabase
    .from('clothes')
    .select(PRODUCT_COLUMNS)
    .order('id', { ascending: false })
    .range(page * HOME_PAGE_SIZE, page * HOME_PAGE_SIZE + HOME_PAGE_SIZE - 1);

  if (category) {
    query = query.eq('sub_category', category);
  }

  const { data, error } = await query;
  if (error) throw error;
  return (data ?? []) as Product[];
}
