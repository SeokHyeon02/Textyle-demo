
export type Product = {
  id: number;
  image_url: string | null;
  brand_name: string | null;
  name: string | null;
  price: number | string | null;
  main_category: string | null;
  sub_category: string | null;
  shop_link: string | null;
};

export const HOME_PAGE_SIZE = 20;

const API_URL = process.env.EXPO_PUBLIC_FASHION_API_URL?.replace(/\/$/, '');

let categoriesCache: string[] | null = null;

// 카테고리(sub_category) 목록.
// Spring API를 통해 카테고리(sub_category) 목록을 가져온다.
export async function fetchCategories(): Promise<string[]> {
  if (categoriesCache) return categoriesCache;

  if (!API_URL) {
    throw new Error('EXPO_PUBLIC_FASHION_API_URL 환경변수가 설정되지 않았습니다.');
  }

  const response = await fetch(
    `${API_URL}/api/products/categories`
  );

  if (!response.ok) {
    throw new Error('카테고리 조회에 실패했습니다.');
  }

  const data: string[] = await response.json();

  categoriesCache = data;
  return data;
}


// Spring API를 통해 카테고리별 상품을 페이지 단위로 가져온다.
// category가 null이면 전체 상품을 조회한다.
// 반환 개수가 HOME_PAGE_SIZE 보다 작으면 더 이상 없음.
export async function fetchProducts(category: string | null, page: number): Promise<Product[]> {
  if (!API_URL) {
    throw new Error('EXPO_PUBLIC_FASHION_API_URL 환경변수가 설정되지 않았습니다.');
  }

  const params = new URLSearchParams({
    page: String(page),
  });

  if (category) {
    params.append('category', category);
  }

  const response = await fetch(
    `${API_URL}/api/products?${params.toString()}`
  );

  if (!response.ok) {
    throw new Error('상품 조회에 실패했습니다.');
  }

  const data = await response.json();

  return data.content ?? [];
}
