import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Dimensions,
  FlatList,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { Href } from 'expo-router';
import { router } from 'expo-router';
import { fetchCategories, fetchProducts, HOME_PAGE_SIZE, Product } from '../../lib/home';

const PLACEHOLDER_IMAGE_URL = 'https://via.placeholder.com/300?text=No+Image';
const H_PADDING = 6;
const COLUMN_GAP = 6;
const CARD_RATIO = 1.4; // 세로가 가로보다 긴 비율
const ITEM_WIDTH = (Dimensions.get('window').width - H_PADDING * 2 - COLUMN_GAP) / 2;
const ITEM_HEIGHT = Math.round(ITEM_WIDTH * CARD_RATIO);

export default function HomeScreen() {
  const [categories, setCategories] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [products, setProducts] = useState<Product[]>([]);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);

  useEffect(() => {
    fetchCategories()
      .then(setCategories)
      .catch((error) => console.warn('카테고리 조회 실패:', error));
  }, []);

  // 선택 카테고리가 바뀌면 첫 페이지부터 다시 로드.
  useEffect(() => {
    let active = true;
    setLoading(true);
    setProducts([]);
    fetchProducts(selectedCategory, 0)
      .then((data) => {
        if (!active) return;
        setProducts(data);
        setPage(0);
        setHasMore(data.length === HOME_PAGE_SIZE);
      })
      .catch((error) => {
        if (!active) return;
        console.warn('상품 조회 실패:', error);
        setHasMore(false);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [selectedCategory]);

  const loadMore = useCallback(async () => {
    if (loading || loadingMore || !hasMore) return;
    const nextPage = page + 1;
    setLoadingMore(true);
    try {
      const data = await fetchProducts(selectedCategory, nextPage);
      setProducts((prev) => [...prev, ...data]);
      setPage(nextPage);
      setHasMore(data.length === HOME_PAGE_SIZE);
    } catch (error) {
      console.warn('추가 상품 조회 실패:', error);
      setHasMore(false);
    } finally {
      setLoadingMore(false);
    }
  }, [loading, loadingMore, hasMore, page, selectedCategory]);

  const getValidImageUrl = (url?: string | null) => {
    if (!url) return PLACEHOLDER_IMAGE_URL;
    const trimmed = url.trim();
    return trimmed.startsWith('//') ? 'https:' + trimmed : trimmed;
  };

  const formatPrice = (price?: number | string | null) => {
    if (price === null || price === undefined || price === '') return '가격 정보 없음';
    const numericPrice = Number(price);
    return Number.isFinite(numericPrice) ? `${numericPrice.toLocaleString()}원` : String(price);
  };

  const openDetail = (item: Product) => {
    router.push({
      pathname: '/product',
      params: {
        id: String(item.id),
        imageUrl: item.image_url ?? '',
        brand: item.brand_name ?? '',
        name: item.name ?? '',
        price: item.price != null ? String(item.price) : '',
        shopLink: item.shop_link ?? '',
      },
    } as Href);
  };

  const renderItem = ({ item }: { item: Product }) => (
    <TouchableOpacity
      style={[styles.card, { width: ITEM_WIDTH }]}
      activeOpacity={0.85}
      onPress={() => openDetail(item)}>
      <Image
        source={{ uri: getValidImageUrl(item.image_url) }}
        style={[styles.cardImage, { width: ITEM_WIDTH, height: ITEM_HEIGHT }]}
        resizeMode="cover"
      />
      <View style={styles.cardOverlay}>
        <Text style={styles.cardBrand} numberOfLines={1}>
          {item.brand_name || '브랜드 미상'}
        </Text>
        <Text style={styles.cardPrice} numberOfLines={1}>
          {formatPrice(item.price)}
        </Text>
      </View>
    </TouchableOpacity>
  );

  const categoryChips: (string | null)[] = [null, ...categories];

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      <View style={styles.categoryBar}>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.categoryContent}>
          {categoryChips.map((cat) => {
            const selected = selectedCategory === cat;
            return (
              <TouchableOpacity
                key={cat ?? '전체'}
                style={[styles.chip, selected && styles.chipSelected]}
                onPress={() => setSelectedCategory(cat)}
                activeOpacity={0.8}>
                <Text style={[styles.chipText, selected && styles.chipTextSelected]}>
                  {cat ?? '전체'}
                </Text>
              </TouchableOpacity>
            );
          })}
        </ScrollView>
      </View>

      <FlatList
        data={products}
        renderItem={renderItem}
        keyExtractor={(item) => String(item.id)}
        numColumns={2}
        columnWrapperStyle={styles.columnWrapper}
        contentContainerStyle={styles.listContent}
        onEndReached={loadMore}
        onEndReachedThreshold={0.5}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={
          loading ? (
            <View style={styles.stateBox}>
              <ActivityIndicator color="#3E6AE1" />
            </View>
          ) : (
            <View style={styles.stateBox}>
              <Text style={styles.stateText}>이 카테고리에 상품이 없어요.</Text>
            </View>
          )
        }
        ListFooterComponent={
          loadingMore ? (
            <View style={styles.footer}>
              <ActivityIndicator color="#8E8E8E" />
            </View>
          ) : null
        }
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  categoryBar: {
    borderBottomWidth: 1,
    borderBottomColor: '#F0F0F0',
  },
  categoryContent: {
    paddingHorizontal: H_PADDING,
    paddingVertical: 12,
    gap: 8,
  },
  chip: {
    paddingHorizontal: 18,
    paddingVertical: 9,
    borderRadius: 20,
    backgroundColor: '#F1F2F3',
  },
  chipSelected: {
    backgroundColor: '#171A20',
  },
  chipText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#393C41',
  },
  chipTextSelected: {
    color: '#FFFFFF',
  },
  listContent: {
    paddingHorizontal: H_PADDING,
    paddingTop: 14,
    paddingBottom: 28,
  },
  columnWrapper: {
    justifyContent: 'space-between',
    marginBottom: COLUMN_GAP,
  },
  card: {
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#F4F4F4',
  },
  cardImage: {
    backgroundColor: '#F4F4F4',
  },
  cardOverlay: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    paddingHorizontal: 12,
    paddingBottom: 12,
    paddingTop: 24,
  },
  cardBrand: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
    textShadowColor: 'rgba(0,0,0,0.6)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
  },
  cardPrice: {
    marginTop: 2,
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '800',
    textShadowColor: 'rgba(0,0,0,0.6)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
  },
  stateBox: {
    paddingTop: 80,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stateText: {
    color: '#5C5E62',
    fontSize: 14,
  },
  footer: {
    paddingVertical: 18,
    alignItems: 'center',
  },
});
