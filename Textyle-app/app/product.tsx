import { Ionicons } from '@expo/vector-icons';
import type { Session } from '@supabase/supabase-js';
import { router, useLocalSearchParams } from 'expo-router';
import React, { useEffect, useRef, useState } from 'react';
import {
  Alert,
  Animated,
  Image,
  Linking,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { addBookmark, fetchBookmarkedIds, removeBookmark } from '../lib/bookmarks';
import { setSearchPresetImage } from '../lib/searchPreset';
import { supabase } from '../supabase';

const PLACEHOLDER_IMAGE_URL = 'https://via.placeholder.com/400?text=No+Image';

export default function ProductDetailScreen() {
  const params = useLocalSearchParams<{
    id?: string;
    imageUrl?: string;
    brand?: string;
    name?: string;
    price?: string;
    shopLink?: string;
  }>();

  const clothId = params.id ? Number(params.id) : null;
  const imageUrl = params.imageUrl || '';
  const brand = params.brand || '';
  const name = params.name || '';
  const priceRaw = params.price || '';
  const shopLink = params.shopLink || '';

  const [session, setSession] = useState<Session | null>(null);
  const [bookmarked, setBookmarked] = useState(false);
  const [toggling, setToggling] = useState(false);

  // backdrop은 제자리에서 페이드인, 카드는 아래에서 위로 슬라이드시켜
  // 회색 배경이 카드와 함께 딸려 올라오는 현상을 막는다.
  const backdropOpacity = useRef(new Animated.Value(0)).current;
  const cardTranslateY = useRef(new Animated.Value(60)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(backdropOpacity, {
        toValue: 1,
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.spring(cardTranslateY, {
        toValue: 0,
        speed: 14,
        bounciness: 6,
        useNativeDriver: true,
      }),
    ]).start();
  }, [backdropOpacity, cardTranslateY]);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setSession(session));
    const { data } = supabase.auth.onAuthStateChange((_event, s) => setSession(s));
    return () => data.subscription.unsubscribe();
  }, []);

  // 현재 상품이 이미 찜되어 있는지 확인해 하트 상태를 맞춘다.
  useEffect(() => {
    const userId = session?.user?.id;
    if (!userId || clothId === null) return;
    let active = true;
    fetchBookmarkedIds(userId)
      .then((ids) => {
        if (active) setBookmarked(ids.includes(clothId));
      })
      .catch((error) => console.warn('찜 상태 조회 실패:', error));
    return () => {
      active = false;
    };
  }, [session?.user?.id, clothId]);

  const getValidImageUrl = (url?: string | null) => {
    if (!url) return PLACEHOLDER_IMAGE_URL;
    const trimmed = url.trim();
    return trimmed.startsWith('//') ? 'https:' + trimmed : trimmed;
  };

  const formatPrice = (price: string) => {
    if (!price) return '가격 정보 없음';
    const numericPrice = Number(price);
    return Number.isFinite(numericPrice) ? `${numericPrice.toLocaleString()}원` : price;
  };

  const close = () => router.back();

  const handleBookmark = async () => {
    const userId = session?.user?.id;
    if (!userId) {
      Alert.alert('알림', '찜하려면 로그인이 필요합니다.');
      return;
    }
    if (clothId === null || toggling) return;

    const wasBookmarked = bookmarked;
    setToggling(true);
    setBookmarked(!wasBookmarked); // 낙관적 업데이트

    try {
      if (wasBookmarked) await removeBookmark(userId, clothId);
      else await addBookmark(userId, clothId);
    } catch (error) {
      setBookmarked(wasBookmarked); // 롤백
      console.error('찜 처리 실패:', error);
      Alert.alert('오류', '찜 처리에 실패했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setToggling(false);
    }
  };

  // 검색하기: 이 상품 이미지를 검색 탭에 자동 첨부하고 검색 탭으로 이동.
  const handleSearch = () => {
    if (imageUrl) setSearchPresetImage(getValidImageUrl(imageUrl));
    router.replace('/');
  };

  // 구매하기: 상품 판매 페이지로 이동.
  const handleBuy = async () => {
    if (!shopLink) {
      Alert.alert('알림', '상품 판매 링크가 없습니다.');
      return;
    }
    let url = shopLink.trim();
    if (url.startsWith('//')) url = 'https:' + url;
    else if (!url.startsWith('http')) url = 'https://' + url;
    try {
      await Linking.openURL(url);
    } catch {
      Alert.alert('오류', '링크를 열 수 없습니다.');
    }
  };

  return (
    <View style={styles.root}>
      <Animated.View style={[styles.backdrop, { opacity: backdropOpacity }]}>
        <Pressable style={StyleSheet.absoluteFill} onPress={close} />
      </Animated.View>

      <Animated.View style={[styles.card, { transform: [{ translateY: cardTranslateY }] }]}>
        <View style={styles.imageWrap}>
          <Image source={{ uri: getValidImageUrl(imageUrl) }} style={styles.image} resizeMode="cover" />
          <TouchableOpacity style={styles.closeBtn} onPress={close} hitSlop={8} activeOpacity={0.8}>
            <Ionicons name="close" size={22} color="#171A20" />
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.info} contentContainerStyle={styles.infoContent}>
          <Text style={styles.brand}>{brand || '브랜드 미상'}</Text>
          {!!name && (
            <Text style={styles.name} numberOfLines={3}>
              {name}
            </Text>
          )}
          <View style={styles.priceRow}>
            <Text style={styles.price}>{formatPrice(priceRaw)}</Text>
            <TouchableOpacity onPress={handleBookmark} disabled={toggling} hitSlop={8} activeOpacity={0.7}>
              <Ionicons
                name={bookmarked ? 'heart' : 'heart-outline'}
                size={30}
                color={bookmarked ? '#EF4444' : '#8E8E8E'}
              />
            </TouchableOpacity>
          </View>
        </ScrollView>

        <View style={styles.bottomBar}>
          <TouchableOpacity style={styles.searchBtn} onPress={handleSearch} activeOpacity={0.85}>
            <Ionicons name="search" size={18} color="#171A20" />
            <Text style={styles.searchBtnText}>검색하기</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.buyBtn} onPress={handleBuy} activeOpacity={0.85}>
            <Text style={styles.buyBtnText}>구매하기</Text>
          </TouchableOpacity>
        </View>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 30,
    paddingVertical: 60,
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  card: {
    height: '82%',
    backgroundColor: '#FFFFFF',
    borderRadius: 20,
    overflow: 'hidden',
  },
  imageWrap: {
    flex: 1,
    backgroundColor: '#F4F4F4',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  closeBtn: {
    position: 'absolute',
    top: 14,
    left: 14,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.15,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 1 },
    elevation: 3,
  },
  info: {
    maxHeight: 200,
    flexGrow: 0,
  },
  infoContent: {
    paddingHorizontal: 20,
    paddingTop: 18,
    paddingBottom: 14,
  },
  brand: {
    fontSize: 19,
    fontWeight: '800',
    color: '#171A20',
  },
  name: {
    marginTop: 6,
    fontSize: 16,
    lineHeight: 23,
    color: '#393C41',
  },
  priceRow: {
    marginTop: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  price: {
    fontSize: 24,
    fontWeight: '800',
    color: '#171A20',
  },
  bottomBar: {
    flexDirection: 'row',
    gap: 10,
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 16,
    borderTopWidth: 1,
    borderTopColor: '#F0F0F0',
    backgroundColor: '#FFFFFF',
  },
  searchBtn: {
    flex: 1,
    flexDirection: 'row',
    gap: 6,
    minHeight: 52,
    borderRadius: 26,
    borderWidth: 1,
    borderColor: '#D0D1D2',
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  searchBtnText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#171A20',
  },
  buyBtn: {
    flex: 1.4,
    minHeight: 52,
    borderRadius: 26,
    backgroundColor: '#243B6E',
    alignItems: 'center',
    justifyContent: 'center',
  },
  buyBtnText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#FFFFFF',
  },
});
