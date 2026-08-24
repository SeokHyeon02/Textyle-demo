import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import type { Session } from '@supabase/supabase-js';
import type { Href } from 'expo-router';
import { router } from 'expo-router';
import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Linking,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { BookmarkRow, fetchBookmarks, removeBookmark } from '../../lib/bookmarks';
import { supabase } from '../../supabase';

const PLACEHOLDER_IMAGE_URL = 'https://via.placeholder.com/200?text=No+Image';

export default function BookmarksScreen() {
  const [session, setSession] = useState<Session | null>(null);
  const [bookmarks, setBookmarks] = useState<BookmarkRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [removingIds, setRemovingIds] = useState<Set<number>>(new Set());
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setSession(session));
    const { data } = supabase.auth.onAuthStateChange((_event, session) => setSession(session));

    return () => data.subscription.unsubscribe();
  }, []);

  const loadBookmarks = useCallback(
    async (mode: 'initial' | 'refresh' = 'initial') => {
      const userId = session?.user?.id;
      if (!userId) {
        setBookmarks([]);
        return;
      }

      if (mode === 'refresh') setRefreshing(true);
      else setLoading(true);
      setErrorMessage(null);

      try {
        const rows = await fetchBookmarks(userId);
        setBookmarks(rows);
      } catch (error) {
        console.error('찜 목록 조회 실패:', error);
        setErrorMessage('찜 목록을 불러오지 못했습니다. 아래로 당겨 새로고침해주세요.');
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [session?.user?.id]
  );

  // 탭에 들어올 때마다 최신 찜 목록을 가져온다 (검색 탭에서 찜한 내용이 바로 반영되도록).
  useFocusEffect(
    useCallback(() => {
      loadBookmarks('initial');
    }, [loadBookmarks])
  );

  const handleRemove = async (clothId: number) => {
    const userId = session?.user?.id;
    if (!userId || removingIds.has(clothId)) return;

    const snapshot = bookmarks;
    setRemovingIds((prev) => new Set(prev).add(clothId));
    setBookmarks((prev) => prev.filter((row) => row.cloth_id !== clothId)); // 낙관적 제거

    try {
      await removeBookmark(userId, clothId);
    } catch (error) {
      setBookmarks(snapshot); // 롤백
      console.error('찜 해제 실패:', error);
      Alert.alert('오류', '찜 해제에 실패했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setRemovingIds((prev) => {
        const next = new Set(prev);
        next.delete(clothId);
        return next;
      });
    }
  };

  const openShopLink = async (link?: string | null) => {
    if (!link) {
      Alert.alert('알림', '상품 링크가 없습니다.');
      return;
    }

    let targetUrl = link.trim();
    if (targetUrl.startsWith('//')) {
      targetUrl = 'https:' + targetUrl;
    } else if (!targetUrl.startsWith('http')) {
      targetUrl = 'https://' + targetUrl;
    }

    try {
      await Linking.openURL(targetUrl);
    } catch {
      Alert.alert('오류', '링크를 열 수 없습니다.');
    }
  };

  // 찜 카드를 터치하면 상품 정보 페이지(모달)를 띄운다.
  const openDetail = (row: BookmarkRow) => {
    const cloth = row.clothes;
    router.push({
      pathname: '/product',
      params: {
        id: String(row.cloth_id),
        imageUrl: cloth?.image_url ?? '',
        brand: cloth?.brand_name ?? '',
        name: cloth?.name ?? '',
        price: cloth?.price != null ? String(cloth.price) : '',
        shopLink: cloth?.shop_link ?? '',
      },
    } as Href);
  };

  const getValidImageUrl = (url?: string | null) => {
    if (!url) return PLACEHOLDER_IMAGE_URL;
    const validUrl = url.trim();
    if (validUrl.startsWith('//')) {
      return 'https:' + validUrl;
    }
    return validUrl;
  };

  const formatCategory = (main?: string | null, sub?: string | null) => {
    const parts = [main, sub].filter(Boolean);
    return parts.length > 0 ? parts.join(' > ') : '카테고리 정보 없음';
  };

  const formatPrice = (price?: number | string | null) => {
    if (price === null || price === undefined || price === '') return '가격 정보 없음';
    const numericPrice = Number(price);
    return Number.isFinite(numericPrice) ? `${numericPrice.toLocaleString()}원` : String(price);
  };

  if (!session) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.centerContainer}>
          <Ionicons name="heart-outline" size={42} color="#5C5E62" />
          <Text style={styles.title}>찜한 상품</Text>
          <Text style={styles.body}>로그인하면 마음에 든 상품을 저장하고 모아볼 수 있어요.</Text>
          <TouchableOpacity style={styles.primaryButton} onPress={() => router.push('/login')}>
            <Text style={styles.primaryButtonText}>로그인하기</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={() => loadBookmarks('refresh')} />
        }>
        <View style={styles.header}>
          <Text style={styles.screenTitle}>찜한 상품</Text>
          <Text style={styles.resultCount}>{bookmarks.length}개 상품</Text>
        </View>

        {loading && bookmarks.length === 0 ? (
          <View style={styles.statePanel}>
            <ActivityIndicator color="#3E6AE1" />
            <Text style={styles.stateBody}>찜한 상품을 불러오는 중...</Text>
          </View>
        ) : errorMessage ? (
          <View style={styles.statePanel}>
            <Ionicons name="alert-circle-outline" size={40} color="#8E8E8E" />
            <Text style={styles.stateBody}>{errorMessage}</Text>
            <TouchableOpacity style={styles.primaryButton} onPress={() => loadBookmarks('initial')}>
              <Text style={styles.primaryButtonText}>다시 시도</Text>
            </TouchableOpacity>
          </View>
        ) : bookmarks.length === 0 ? (
          <View style={styles.statePanel}>
            <Ionicons name="heart-outline" size={42} color="#8E8E8E" />
            <Text style={styles.emptyTitle}>아직 찜한 상품이 없어요.</Text>
            <Text style={styles.stateBody}>검색 결과에서 하트를 누르면 여기에 모아둘 수 있어요.</Text>
            <TouchableOpacity style={styles.primaryButton} onPress={() => router.push('/')}>
              <Text style={styles.primaryButtonText}>상품 검색하러 가기</Text>
            </TouchableOpacity>
          </View>
        ) : (
          bookmarks.map((row) => {
            const cloth = row.clothes;
            const isRemoving = removingIds.has(row.cloth_id);

            return (
              <TouchableOpacity
                key={row.cloth_id}
                style={styles.resultCard}
                activeOpacity={0.85}
                onPress={() => openDetail(row)}>
                <View style={styles.resultImageWrap}>
                  <Image
                    source={{ uri: getValidImageUrl(cloth?.image_url) }}
                    style={styles.resultImage}
                    resizeMode="cover"
                  />
                  <TouchableOpacity
                    style={styles.heartButton}
                    onPress={() => handleRemove(row.cloth_id)}
                    disabled={isRemoving}
                    hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                    activeOpacity={0.8}>
                    <Ionicons name="heart" size={20} color="#EF4444" />
                  </TouchableOpacity>
                </View>
                <View style={styles.resultInfo}>
                  <View style={styles.infoSection}>
                    <Text style={styles.infoLabel}>브랜드</Text>
                    <Text style={styles.resultBrand} numberOfLines={1}>
                      {cloth?.brand_name || '브랜드 정보 없음'}
                    </Text>
                  </View>
                  <View style={styles.infoDivider} />
                  <View style={styles.infoSection}>
                    <Text style={styles.infoLabel}>제품명</Text>
                    <Text style={styles.resultName} numberOfLines={2}>
                      {cloth?.name || '상품명 정보 없음'}
                    </Text>
                  </View>
                  <View style={styles.infoDivider} />
                  <View style={styles.infoGrid}>
                    <View style={styles.infoGridItem}>
                      <Text style={styles.infoLabel}>카테고리</Text>
                      <Text style={styles.resultCategory} numberOfLines={1}>
                        {formatCategory(cloth?.main_category, cloth?.sub_category)}
                      </Text>
                    </View>
                    <View style={styles.infoGridItem}>
                      <Text style={styles.infoLabel}>가격</Text>
                      <Text style={styles.resultPrice}>{formatPrice(cloth?.price)}</Text>
                    </View>
                  </View>
                  <View style={styles.infoDivider} />
                  <TouchableOpacity
                    style={styles.linkButton}
                    onPress={() => openShopLink(cloth?.shop_link)}>
                    <Text style={styles.linkButtonText}>상품 보러가기</Text>
                  </TouchableOpacity>
                </View>
              </TouchableOpacity>
            );
          })
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 28,
    backgroundColor: '#FFFFFF',
  },
  content: {
    paddingHorizontal: 20,
    paddingTop: 28,
    paddingBottom: 32,
    flexGrow: 1,
  },
  header: {
    width: '100%',
    maxWidth: 430,
    alignSelf: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  screenTitle: {
    fontSize: 26,
    fontWeight: '600',
    color: '#171A20',
    textAlign: 'center',
  },
  resultCount: {
    marginTop: 6,
    color: '#5C5E62',
    fontSize: 14,
    textAlign: 'center',
  },
  title: {
    marginTop: 16,
    fontSize: 24,
    fontWeight: '600',
    color: '#171A20',
    textAlign: 'center',
  },
  body: {
    marginTop: 10,
    fontSize: 15,
    lineHeight: 22,
    color: '#393C41',
    textAlign: 'center',
  },
  statePanel: {
    minHeight: 360,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 8,
    backgroundColor: '#F4F4F4',
    paddingHorizontal: 24,
  },
  emptyTitle: {
    marginTop: 16,
    color: '#171A20',
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
  },
  stateBody: {
    marginTop: 8,
    color: '#5C5E62',
    fontSize: 14,
    lineHeight: 20,
    textAlign: 'center',
  },
  primaryButton: {
    minWidth: 200,
    minHeight: 50,
    borderRadius: 6,
    backgroundColor: '#3E6AE1',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
    marginTop: 20,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  resultCard: {
    flexDirection: 'row',
    width: '100%',
    maxWidth: 430,
    alignSelf: 'center',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#EEEEEE',
    backgroundColor: '#FFFFFF',
    padding: 12,
    marginBottom: 14,
  },
  resultImageWrap: {
    width: 116,
    height: 148,
    marginRight: 14,
  },
  resultImage: {
    width: 116,
    height: 148,
    borderRadius: 6,
    backgroundColor: '#F4F4F4',
  },
  heartButton: {
    position: 'absolute',
    top: 6,
    right: 6,
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(0,0,0,0.38)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  resultInfo: {
    flex: 1,
    minHeight: 148,
  },
  infoSection: {
    paddingBottom: 7,
  },
  infoDivider: {
    height: 1,
    backgroundColor: '#EEEEEE',
    marginBottom: 7,
  },
  infoLabel: {
    color: '#8E8E8E',
    fontSize: 11,
    lineHeight: 15,
    marginBottom: 2,
  },
  infoGrid: {
    flexDirection: 'row',
    gap: 10,
    paddingBottom: 7,
  },
  infoGridItem: {
    flex: 1,
  },
  resultBrand: {
    color: '#393C41',
    fontSize: 14,
    fontWeight: '600',
  },
  resultName: {
    color: '#171A20',
    fontSize: 15,
    fontWeight: '600',
    lineHeight: 20,
  },
  resultCategory: {
    color: '#5C5E62',
    fontSize: 12,
    lineHeight: 16,
  },
  resultPrice: {
    color: '#171A20',
    fontSize: 15,
    fontWeight: '600',
  },
  linkButton: {
    alignSelf: 'flex-start',
    justifyContent: 'center',
  },
  linkButtonText: {
    color: '#3E6AE1',
    fontSize: 14,
    fontWeight: '600',
  },
});
