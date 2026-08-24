import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import type { Session } from '@supabase/supabase-js';
import * as FileSystem from 'expo-file-system/legacy';
import * as ImagePicker from 'expo-image-picker';
import type { Href } from 'expo-router';
import { router } from 'expo-router';
import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Linking,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { addBookmark, fetchBookmarkedIds, removeBookmark } from '../../lib/bookmarks';
import { consumeSearchPresetImage } from '../../lib/searchPreset';
import { supabase } from '../../supabase';

const FASHION_API_URL = process.env.EXPO_PUBLIC_FASHION_API_URL?.replace(/\/$/, '');
const PLACEHOLDER_IMAGE_URL = 'https://via.placeholder.com/200?text=No+Image';

type SearchResult = {
  id?: number | null;
  image_url?: string | null;
  main_category?: string | null;
  sub_category?: string | null;
  brand_name?: string | null;
  name?: string | null;
  price?: number | string | null;
  similarity?: number | null;
  shop_link?: string | null;
  _ranking?: RankingInfo;
};

type RankingInfo = {
  final_score?: number;
  base_similarity?: number;
  color_adjustment?: number;
  category_bonus?: number;
  sub_category_bonus?: number;
  tone_adjustment?: number;
  design_adjustment?: number;
  design_matches?: string[];
  design_conflicts?: string[];
  candidate_color?: string;
  candidate_denim_tone?: string;
  exclude_reason?: string;
};

type SearchTiming = {
  total_ms?: number;
  validate_ms?: number;
  query_analysis_ms?: number;
  gemini_ms?: number;
  dino_sam_ms?: number;
  embedding_ms?: number;
  rpc_ms?: number;
  rerank_ms?: number;
  [key: string]: number | undefined;
};

type SearchMetadata = {
  enhanced_query?: string;
  color_extracted?: {
    color?: string;
    detailed_color?: string;
    confidence?: string;
    pattern?: string;
  };
  intent?: {
    color_mode?: string;
    color?: string;
    design?: string;
    reasoning?: string;
  };
  query_image_attributes?: {
    main_categories?: string[];
    sub_categories?: string[];
    image_preprocess_source?: string;
    denim_tone?: string;
    [key: string]: unknown;
  };
  search_warnings?: string[];
  timing?: SearchTiming;
};

const COLOR_MODE_LABELS: Record<string, string> = {
  same: '같은 색',
  different: '다른 색',
  target: '특정 색',
  ignore: '색상 무관',
};

const WARNING_LABELS: Record<string, string> = {
  uploaded_image_color_not_detected: '⚠️ 이미지 색상을 안정적으로 찾지 못했습니다',
  uploaded_image_color_low_confidence: '⚠️ 색상 추출 신뢰도가 낮습니다',
  design_similarity_uses_image_embedding_only: 'ℹ️ 디자인 유사도는 이미지 특징 기반입니다',
};

const LOADING_MESSAGES = [
  '🔍 이미지 분석 중...',
  '🧠 검색 의도 해석 중...',
  '👕 유사 상품 검색 중...',
  '⏳ 정밀 검색 중입니다. 조금만 기다려주세요.',
];

const getRankingLabels = (ranking?: RankingInfo): string[] => {
  if (!ranking) return [];
  const labels: string[] = [];
  if ((ranking.category_bonus ?? 0) > 0 || (ranking.sub_category_bonus ?? 0) > 0) {
    labels.push('📂 카테고리 일치');
  }
  if ((ranking.color_adjustment ?? 0) > 0.05) {
    labels.push('🎨 색상 일치');
  }
  if ((ranking.tone_adjustment ?? 0) > 0) {
    labels.push('🔵 데님 톤 일치');
  }
  if (ranking.design_matches && ranking.design_matches.length > 0) {
    labels.push(`✂️ 디자인 유사 (${ranking.design_matches.join(', ')})`);
  }
  if ((ranking.color_adjustment ?? 0) < -0.10) {
    labels.push('⚠️ 색상 불일치');
  }
  if (ranking.exclude_reason) {
    labels.push('⛔ 필터 충돌');
  }
  return labels;
};

/*
const formatTiming = (value?: number) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value >= 1000 ? `${(value / 1000).toFixed(1)}초` : `${Math.round(value)}ms`;
};

const getTimingRows = (timing?: SearchTiming) => {
  if (!timing) return [];

  return [
    ['검증', timing.validate_ms],
    ['쿼리 해석', timing.query_analysis_ms],
    ['Gemini', timing.gemini_ms],
    ['DINO/SAM', timing.dino_sam_ms],
    ['임베딩', timing.embedding_ms],
    ['DB 검색', timing.rpc_ms],
    ['재정렬', timing.rerank_ms],
  ]
    .map(([label, value]) => ({ label: label as string, value: formatTiming(value as number | undefined) }))
    .filter(row => row.value);
};
*/

export default function SearchScreen() {
  const [session, setSession] = useState<Session | null>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [searchText, setSearchText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [searchMetadata, setSearchMetadata] = useState<SearchMetadata | null>(null);
  const [loadingStage, setLoadingStage] = useState(0);
  const [useGroundingDino, setUseGroundingDino] = useState(false);
  const [bookmarkedIds, setBookmarkedIds] = useState<Set<number>>(new Set());
  const [togglingIds, setTogglingIds] = useState<Set<number>>(new Set());

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setSession(session));
    const { data } = supabase.auth.onAuthStateChange((_event, session) => setSession(session));

    return () => data.subscription.unsubscribe();
  }, []);

  useFocusEffect(
    useCallback(() => {
      const presetUrl = consumeSearchPresetImage();
      if (presetUrl) {
        setImageUri(presetUrl);
        setSelectedImage(null);
        setSearchText('');
        setSearchResults([]);
        setHasSearched(false);
        setErrorMessage(null);
        setSearchMetadata(null);
      }
    }, [])
  );

  useEffect(() => {
    if (!session?.user?.id || searchResults.length === 0) {
      setBookmarkedIds(new Set());
      return;
    }

    let active = true;
    fetchBookmarkedIds(session.user.id)
      .then((ids) => {
        if (active) setBookmarkedIds(new Set(ids));
      })
      .catch((error) => console.warn('찜 목록 조회 실패:', error));

    return () => {
      active = false;
    };
  }, [session?.user?.id, searchResults]);

  const toggleBookmark = async (item: SearchResult) => {
    const userId = session?.user?.id;
    if (!userId) {
      Alert.alert('알림', '찜하려면 로그인이 필요합니다.');
      return;
    }
    if (item.id === null || item.id === undefined) {
      Alert.alert('알림', '이 상품은 찜할 수 없습니다.');
      return;
    }

    const clothId = item.id;
    if (togglingIds.has(clothId)) return;

    const wasBookmarked = bookmarkedIds.has(clothId);
    setTogglingIds((prev) => new Set(prev).add(clothId));
    setBookmarkedIds((prev) => {
      const next = new Set(prev);
      if (wasBookmarked) next.delete(clothId);
      else next.add(clothId);
      return next;
    });

    try {
      if (wasBookmarked) await removeBookmark(userId, clothId);
      else await addBookmark(userId, clothId);
    } catch (error) {
      setBookmarkedIds((prev) => {
        const next = new Set(prev);
        if (wasBookmarked) next.add(clothId);
        else next.delete(clothId);
        return next;
      });
      console.error('찜 처리 실패:', error);
      Alert.alert('오류', '찜 처리에 실패했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setTogglingIds((prev) => {
        const next = new Set(prev);
        next.delete(clothId);
        return next;
      });
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      quality: 1,
      preferredAssetRepresentationMode: ImagePicker.UIImagePickerPreferredAssetRepresentationMode.Automatic,
    });

    if (!result.canceled) {
      const asset = result.assets[0];
      setImageUri(asset.uri);
      setSelectedImage(asset);
      setErrorMessage(null);
    }
  };

  const searchClothes = async () => {
    if (!imageUri) {
      setErrorMessage('검색할 사진을 먼저 선택해주세요.');
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setSearchMetadata(null);
    setLoadingStage(0);
    const stageTimer = setInterval(() => setLoadingStage(prev => prev + 1), 4000);

    try {
      if (!FASHION_API_URL) {
        throw new Error('EXPO_PUBLIC_FASHION_API_URL 환경변수가 설정되지 않았습니다.');
      }

      const formData = new FormData();
      let uploadUri = selectedImage?.uri || imageUri;
      let fileName = selectedImage?.fileName || 'photo.jpg';
      let mimeType = selectedImage?.mimeType || 'image/jpeg';

      if (uploadUri && /^https?:\/\//.test(uploadUri)) {
        const target = `${FileSystem.cacheDirectory}search-${Date.now()}.jpg`;
        const downloaded = await FileSystem.downloadAsync(uploadUri, target);
        uploadUri = downloaded.uri;
        fileName = 'photo.jpg';
        mimeType = 'image/jpeg';
      }

      formData.append('file', {
        uri: uploadUri,
        name: fileName,
        type: mimeType,
      } as any);

      formData.append('query', searchText.trim());
      formData.append('use_grounding_dino', useGroundingDino ? 'true' : 'false');

      const response = await fetch(`${FASHION_API_URL}/search`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || '검색 요청을 처리하지 못했습니다.');
      }

      const data = await response.json();
      setSearchResults(Array.isArray(data.results) ? data.results : []);
      setSearchMetadata({
        enhanced_query: data.enhanced_query,
        color_extracted: data.color_extracted,
        intent: data.intent,
        query_image_attributes: data.query_image_attributes,
        search_warnings: data.search_warnings,
        timing: data.timing,
      });
      setHasSearched(true);
    } catch (error) {
      console.error('검색 에러:', error);
      const message = error instanceof Error
        ? error.message
        : '서버에 연결할 수 없습니다. FastAPI 서버와 API 주소를 확인해주세요.';
      setErrorMessage(message);
    } finally {
      clearInterval(stageTimer);
      setLoadingStage(0);
      setIsLoading(false);
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

  const openDetail = (item: SearchResult) => {
    router.push({
      pathname: '/product',
      params: {
        id: item.id != null ? String(item.id) : '',
        imageUrl: item.image_url ?? '',
        brand: item.brand_name ?? '',
        name: item.name ?? '',
        price: item.price != null ? String(item.price) : '',
        shopLink: item.shop_link ?? '',
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

  const formatCategory = (item: SearchResult) => {
    const parts = [item.main_category, item.sub_category].filter(Boolean);
    return parts.length > 0 ? parts.join(' > ') : '카테고리 정보 없음';
  };

  const formatPrice = (price?: number | string | null) => {
    if (price === null || price === undefined || price === '') return '가격 정보 없음';
    const numericPrice = Number(price);
    return Number.isFinite(numericPrice) ? `${numericPrice.toLocaleString()}원` : String(price);
  };

  const resetResults = () => {
    setImageUri(null);
    setSelectedImage(null);
    setSearchText('');
    setSearchResults([]);
    setHasSearched(false);
    setErrorMessage(null);
    setSearchMetadata(null);
    setLoadingStage(0);
    setUseGroundingDino(false);
  };

  if (!session) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.centerContainer}>
          <Text style={styles.brandTitle}>Textyle</Text>
          <Text style={styles.subtitle}>이미지와 문장으로 비슷한 옷을 찾아보세요.</Text>
          <TouchableOpacity style={styles.primaryButton} onPress={() => router.push('/login')}>
            <Text style={styles.primaryButtonText}>로그인하고 검색하기</Text>
          </TouchableOpacity>
          <Text style={styles.helperText}>가입 후 이미지 검색과 결과 확인을 사용할 수 있습니다.</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (hasSearched) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.resultContent}>
          <View style={styles.resultHeader}>
            <View>
              <Text style={styles.screenTitle}>검색 결과</Text>
              <Text style={styles.resultCount}>{searchResults.length}개 상품을 비교해보세요.</Text>
            </View>
            <TouchableOpacity style={styles.secondaryButton} onPress={resetResults}>
              <Text style={styles.secondaryButtonText}>다시 검색</Text>
            </TouchableOpacity>
          </View>

          {searchResults.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="search-outline" size={40} color="#8E8E8E" />
              <Text style={styles.emptyTitle}>조건에 맞는 상품을 찾지 못했어요.</Text>
              <Text style={styles.emptyBody}>다른 사진이나 더 넓은 조건으로 다시 검색해보세요.</Text>
              <TouchableOpacity style={styles.primaryButton} onPress={resetResults}>
                <Text style={styles.primaryButtonText}>검색 화면으로 돌아가기</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <>
              {searchMetadata && (
                <View style={styles.metadataCard}>
                  <Text style={styles.metadataTitle}>🔍 검색 해석</Text>
                  {searchMetadata.enhanced_query && (
                    <Text style={styles.metadataRow}>
                      서버 해석: {searchMetadata.enhanced_query}
                    </Text>
                  )}
                  {searchMetadata.intent?.color_mode && searchMetadata.intent.color_mode !== 'ignore' && (
                    <Text style={styles.metadataRow}>
                      색상 조건: {COLOR_MODE_LABELS[searchMetadata.intent.color_mode] || searchMetadata.intent.color_mode}
                      {searchMetadata.intent.color ? ` (${searchMetadata.intent.color})` : ''}
                    </Text>
                  )}
                  {searchMetadata.color_extracted?.color && (
                    <Text style={styles.metadataRow}>
                      이미지 색상: {searchMetadata.color_extracted.detailed_color || searchMetadata.color_extracted.color}
                      {searchMetadata.color_extracted.confidence ? ` (신뢰도 ${searchMetadata.color_extracted.confidence})` : ''}
                    </Text>
                  )}
                  {searchMetadata.query_image_attributes?.main_categories?.[0] && (
                    <Text style={styles.metadataRow}>
                      카테고리: {searchMetadata.query_image_attributes.main_categories[0]}
                      {searchMetadata.query_image_attributes.sub_categories?.[0] ? ` > ${searchMetadata.query_image_attributes.sub_categories[0]}` : ''}
                    </Text>
                  )}
                  {searchMetadata.query_image_attributes?.image_preprocess_source && (
                    <Text style={styles.metadataRow}>
                      이미지 분석: {searchMetadata.query_image_attributes.image_preprocess_source === 'groundingdino_sam' ? '정밀 분석' : searchMetadata.query_image_attributes.image_preprocess_source}
                    </Text>
                  )}
                  {/*
                  {searchMetadata.timing?.total_ms != null && (
                    <View style={styles.timingBlock}>
                      <Text style={styles.metadataRow}>
                        응답 시간: {formatTiming(searchMetadata.timing.total_ms)}
                      </Text>
                      <View style={styles.timingGrid}>
                        {getTimingRows(searchMetadata.timing).map(row => (
                          <View key={row.label} style={styles.timingPill}>
                            <Text style={styles.timingLabel}>{row.label}</Text>
                            <Text style={styles.timingValue}>{row.value}</Text>
                          </View>
                        ))}
                      </View>
                    </View>
                  )}
                  */}
                  {searchMetadata.search_warnings && searchMetadata.search_warnings.length > 0 && (
                    <View style={styles.warningsContainer}>
                      {searchMetadata.search_warnings.map((warning, idx) => (
                        <Text key={idx} style={styles.warningText}>
                          {WARNING_LABELS[warning] || warning}
                        </Text>
                      ))}
                    </View>
                  )}
                </View>
              )}

              {searchResults.map((item, index) => {
                const rankingLabels = getRankingLabels(item._ranking);
                const canBookmark = item.id !== null && item.id !== undefined;
                const isBookmarked = canBookmark && bookmarkedIds.has(item.id as number);
                const isToggling = canBookmark && togglingIds.has(item.id as number);

                return (
                  <TouchableOpacity
                    key={`${item.id ?? item.name ?? 'result'}-${index}`}
                    style={styles.resultCard}
                    activeOpacity={0.88}
                    onPress={() => openDetail(item)}>
                    <View style={styles.resultImageFrame}>
                      <Image
                        source={{ uri: getValidImageUrl(item.image_url) }}
                        style={styles.resultImage}
                        resizeMode="contain"
                      />
                      <View style={styles.rankBadge}>
                        <Text style={styles.rankBadgeText}>#{index + 1}</Text>
                      </View>
                      {canBookmark && (
                        <TouchableOpacity
                          style={styles.heartButton}
                          onPress={() => toggleBookmark(item)}
                          disabled={isToggling}
                          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                          activeOpacity={0.8}>
                          <Ionicons
                            name={isBookmarked ? 'heart' : 'heart-outline'}
                            size={21}
                            color={isBookmarked ? '#EF4444' : '#FFFFFF'}
                          />
                        </TouchableOpacity>
                      )}
                    </View>
                    <View style={styles.resultInfo}>
                      <View style={styles.resultTopLine}>
                        <Text style={styles.resultBrand} numberOfLines={1}>
                          {item.brand_name || '브랜드 정보 없음'}
                        </Text>
                        <Text style={styles.resultPrice}>{formatPrice(item.price)}</Text>
                      </View>
                      <Text style={styles.resultName} numberOfLines={2}>
                        {item.name || '상품명 정보 없음'}
                      </Text>
                      <Text style={styles.resultCategory} numberOfLines={1}>
                        {formatCategory(item)}
                      </Text>
                      {rankingLabels.length > 0 && (
                        <View style={styles.rankingBlock}>
                          <Text style={styles.rankingTitle}>추천 이유</Text>
                          <View style={styles.rankingLabelsContainer}>
                            {rankingLabels.map((label, labelIdx) => (
                              <Text key={labelIdx} style={styles.rankingLabel}>{label}</Text>
                            ))}
                          </View>
                        </View>
                      )}
                      <TouchableOpacity
                        style={styles.resultActionButton}
                        onPress={() => openShopLink(item.shop_link)}>
                        <Text style={styles.resultActionButtonText}>상품 보러가기</Text>
                        <View style={styles.resultActionIcon}>
                          <Ionicons name="arrow-forward" size={15} color="#FFFFFF" />
                        </View>
                      </TouchableOpacity>
                    </View>
                  </TouchableOpacity>
                );
              })}

              <TouchableOpacity style={styles.bottomReturnButton} onPress={resetResults}>
                <Text style={styles.primaryButtonText}>검색 화면으로 돌아가기</Text>
              </TouchableOpacity>
            </>
          )}
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.entryContent} keyboardShouldPersistTaps="handled">
        <View style={styles.searchHomeHeader}>
          <View style={styles.searchHomeTitleBlock}>
            <Text style={styles.searchGreeting}>반가워요!</Text>
            <Text style={styles.searchHomeTitle}>
              원하는 의류를 찾아주는 Textyle입니다!.{'\n'}
              이미지와 문장을 입력하면 비슷한 의류를 찾아드려요!
            </Text>
          </View>
        </View>

        <View style={styles.searchHeroCard}>
          <View style={styles.searchHeroImageArea}>
            <TouchableOpacity style={styles.searchHeroImagePicker} onPress={pickImage} activeOpacity={0.88}>
              {imageUri ? (
                <Image source={{ uri: imageUri }} style={styles.searchHeroPreview} resizeMode="contain" />
              ) : (
                <View style={styles.searchHeroIllustration}>
                  <Ionicons name="shirt-outline" size={104} color="#8E8E8E" />
                  <Text style={styles.searchHeroHint}>이미지를 선택하세요</Text>
                  <View style={styles.sparkleOne}>
                    <Ionicons name="sparkles" size={18} color="#3E6AE1" />
                  </View>
                  <View style={styles.sparkleTwo}>
                    <Ionicons name="sparkles" size={16} color="#3E6AE1" />
                  </View>
                </View>
              )}
            </TouchableOpacity>
          </View>
        </View>

        <TextInput
          style={styles.textInput}
          placeholder="예: 회색 와이드 데님 팬츠"
          placeholderTextColor="#8E8E8E"
          value={searchText}
          onChangeText={setSearchText}
          returnKeyType="search"
        />

        <TouchableOpacity
          style={[styles.analysisToggle, useGroundingDino && styles.analysisToggleActive]}
          onPress={() => setUseGroundingDino(prev => !prev)}
          activeOpacity={0.82}>
          <View style={styles.analysisToggleTextBlock}>
            <Text style={styles.analysisToggleTitle}>정밀 분석</Text>
            <Text style={styles.analysisToggleBody}>
              켜면 옷 영역을 먼저 분리해서 검색합니다. 느리고 결과가 달라질 수 있어요.
            </Text>
          </View>
          <View style={[styles.toggleTrack, useGroundingDino && styles.toggleTrackActive]}>
            <View style={[styles.toggleThumb, useGroundingDino && styles.toggleThumbActive]}>
              {useGroundingDino && <Ionicons name="checkmark" size={13} color="#3E6AE1" />}
            </View>
          </View>
        </TouchableOpacity>

        {errorMessage ? (
          <Text style={styles.errorText}>{errorMessage}</Text>
        ) : (
          <Text style={styles.helperText}>검색어는 선택 사항입니다. 사진만으로도 검색할 수 있어요.</Text>
        )}

        <TouchableOpacity
          style={[styles.searchHeroButton, isLoading && styles.disabledButton]}
          onPress={searchClothes}
          disabled={isLoading}
          activeOpacity={0.86}>
          {isLoading ? (
            <View style={styles.loadingRow}>
              <ActivityIndicator color="#fff" />
              <Text style={styles.searchHeroButtonText}>{LOADING_MESSAGES[Math.min(loadingStage, LOADING_MESSAGES.length - 1)]}</Text>
            </View>
          ) : (
            <Text style={styles.searchHeroButtonText}>검색하기</Text>
          )}
        </TouchableOpacity>

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
  entryContent: {
    flexGrow: 1,
    paddingHorizontal: 24,
    paddingTop: 64,
    paddingBottom: 32,
  },
  resultContent: {
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 32,
  },
  headerBlock: {
    width: '100%',
    maxWidth: 430,
    alignItems: 'center',
    marginBottom: 24,
  },
  brandTitle: {
    fontSize: 32,
    fontWeight: '600',
    color: '#171A20',
    marginBottom: 10,
  },
  screenTitle: {
    fontSize: 26,
    fontWeight: '600',
    color: '#171A20',
    textAlign: 'center',
  },
  subtitle: {
    marginTop: 8,
    fontSize: 16,
    lineHeight: 23,
    color: '#393C41',
    textAlign: 'center',
    width: '100%',
  },
  helperText: {
    marginTop: 12,
    width: '100%',
    maxWidth: 360,
    fontSize: 13,
    lineHeight: 19,
    color: '#5C5E62',
    textAlign: 'center',
  },
  errorText: {
    marginTop: 12,
    fontSize: 13,
    lineHeight: 19,
    color: '#B42318',
  },
  searchHomeHeader: {
    width: '100%',
    maxWidth: 430,
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 16,
    marginBottom: 18,
  },
  searchHomeTitleBlock: {
    flex: 1,
  },
  searchGreeting: {
    color: '#171A20',
    fontSize: 22,
    lineHeight: 29,
    fontWeight: '600',
    marginBottom: 4,
  },
  searchHomeTitle: {
    color: '#171A20',
    fontSize: 17,
    lineHeight: 25,
    fontWeight: '600',
  },
  searchHeroCard: {
    width: '100%',
    maxWidth: 430,
    alignSelf: 'center',
    borderRadius: 8,
    backgroundColor: '#F4F4F4',
    alignItems: 'center',
    padding: 16,
    marginBottom: 18,
  },
  searchHeroImageArea: {
    width: '100%',
    height: 190,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  searchHeroImagePicker: {
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  searchHeroPreview: {
    width: '100%',
    height: '100%',
    borderRadius: 8,
  },
  searchHeroIllustration: {
    width: 178,
    height: 132,
    alignItems: 'center',
    justifyContent: 'center',
  },
  searchHeroHint: {
    marginTop: -8,
    color: '#8E8E8E',
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '500',
  },
  sparkleOne: {
    position: 'absolute',
    left: 12,
    top: 30,
  },
  sparkleTwo: {
    position: 'absolute',
    right: 14,
    bottom: 30,
  },
  searchHeroButton: {
    width: '100%',
    maxWidth: 430,
    alignSelf: 'center',
    minHeight: 56,
    borderRadius: 8,
    backgroundColor: '#3E6AE1',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 20,
  },
  searchHeroButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    lineHeight: 24,
    fontWeight: '600',
  },
  textInput: {
    width: '100%',
    maxWidth: 430,
    height: 56,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#EEEEEE',
    backgroundColor: '#FFFFFF',
    color: '#171A20',
    paddingHorizontal: 14,
    paddingVertical: 14,
    fontSize: 15,
    lineHeight: 21,
    textAlignVertical: 'center',
  },
  analysisToggle: {
    width: '100%',
    maxWidth: 430,
    minHeight: 74,
    marginTop: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#EEEEEE',
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 14,
    paddingVertical: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  analysisToggleActive: {
    borderColor: '#AFC2FF',
    backgroundColor: '#F6F8FF',
  },
  analysisToggleTextBlock: {
    flex: 1,
    minWidth: 0,
  },
  analysisToggleTitle: {
    fontSize: 15,
    lineHeight: 21,
    fontWeight: '600',
    color: '#171A20',
  },
  analysisToggleBody: {
    marginTop: 3,
    fontSize: 12,
    lineHeight: 17,
    color: '#5C5E62',
  },
  toggleTrack: {
    width: 44,
    height: 26,
    borderRadius: 13,
    backgroundColor: '#D9D9D9',
    padding: 3,
    justifyContent: 'center',
  },
  toggleTrackActive: {
    backgroundColor: '#3E6AE1',
  },
  toggleThumb: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  toggleThumbActive: {
    transform: [{ translateX: 18 }],
  },
  primaryButton: {
    width: '100%',
    maxWidth: 430,
    minHeight: 52,
    borderRadius: 6,
    backgroundColor: '#3E6AE1',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
    marginTop: 20,
  },
  disabledButton: {
    opacity: 0.72,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  resultHeader: {
    width: '100%',
    maxWidth: 430,
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 16,
    marginBottom: 18,
  },
  resultCount: {
    marginTop: 6,
    color: '#5C5E62',
    fontSize: 14,
    lineHeight: 20,
  },
  secondaryButton: {
    minHeight: 38,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#D0D1D2',
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 13,
  },
  secondaryButtonText: {
    color: '#393C41',
    fontSize: 13,
    fontWeight: '600',
  },
  emptyState: {
    minHeight: 420,
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
  emptyBody: {
    marginTop: 8,
    color: '#5C5E62',
    fontSize: 14,
    lineHeight: 20,
    textAlign: 'center',
  },
  metadataCard: {
    width: '100%',
    maxWidth: 430,
    alignSelf: 'center',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#EEEEEE',
    backgroundColor: '#F4F4F4',
    padding: 14,
    marginBottom: 16,
  },
  metadataTitle: {
    color: '#171A20',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
  },
  metadataRow: {
    color: '#393C41',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 4,
  },
  timingBlock: {
    marginTop: 4,
  },
  timingGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 8,
  },
  timingPill: {
    borderRadius: 6,
    backgroundColor: '#F4F4F4',
    paddingHorizontal: 8,
    paddingVertical: 6,
  },
  timingLabel: {
    color: '#5C5E62',
    fontSize: 11,
    lineHeight: 14,
  },
  timingValue: {
    color: '#171A20',
    fontSize: 12,
    fontWeight: '600',
    lineHeight: 16,
  },
  warningsContainer: {
    marginTop: 10,
    gap: 6,
  },
  warningText: {
    color: '#B54708',
    fontSize: 12,
    lineHeight: 17,
  },
  resultCard: {
    width: '100%',
    maxWidth: 430,
    alignSelf: 'center',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#EEEEEE',
    backgroundColor: '#FFFFFF',
    overflow: 'hidden',
    marginBottom: 18,
  },
  resultImageFrame: {
    width: '100%',
    height: 286,
    backgroundColor: '#FAFAFA',
    borderBottomWidth: 1,
    borderBottomColor: '#EEEEEE',
    paddingHorizontal: 12,
    paddingVertical: 12,
  },
  resultImage: {
    width: '100%',
    height: '100%',
  },
  rankBadge: {
    position: 'absolute',
    left: 12,
    top: 12,
    minHeight: 30,
    borderRadius: 4,
    backgroundColor: 'rgba(255,255,255,0.94)',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  rankBadgeText: {
    color: '#171A20',
    fontSize: 13,
    fontWeight: '600',
  },
  heartButton: {
    position: 'absolute',
    right: 12,
    top: 12,
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: 'rgba(0,0,0,0.42)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  resultInfo: {
    padding: 14,
  },
  resultTopLine: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    marginBottom: 8,
  },
  resultBrand: {
    flex: 1,
    color: '#5C5E62',
    fontSize: 14,
    fontWeight: '600',
  },
  resultName: {
    color: '#171A20',
    fontSize: 16,
    fontWeight: '600',
    lineHeight: 22,
  },
  resultCategory: {
    marginTop: 6,
    color: '#5C5E62',
    fontSize: 13,
    lineHeight: 18,
  },
  resultPrice: {
    color: '#171A20',
    fontSize: 15,
    fontWeight: '600',
  },
  rankingBlock: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#EEEEEE',
  },
  rankingTitle: {
    color: '#5C5E62',
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '600',
  },
  rankingLabelsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 8,
  },
  rankingLabel: {
    borderRadius: 4,
    backgroundColor: '#EEF3FF',
    color: '#3451B2',
    fontSize: 11,
    lineHeight: 15,
    paddingHorizontal: 6,
    paddingVertical: 3,
  },
  resultActionButton: {
    minHeight: 46,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#D0D1D2',
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexDirection: 'row',
    marginTop: 14,
    paddingLeft: 14,
    paddingRight: 8,
  },
  resultActionButtonText: {
    color: '#171A20',
    fontSize: 15,
    fontWeight: '600',
  },
  resultActionIcon: {
    width: 30,
    height: 30,
    borderRadius: 4,
    backgroundColor: '#3E6AE1',
    alignItems: 'center',
    justifyContent: 'center',
  },
  bottomReturnButton: {
    width: '100%',
    maxWidth: 430,
    minHeight: 52,
    borderRadius: 6,
    backgroundColor: '#3E6AE1',
    alignSelf: 'center',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
    marginTop: 10,
    marginBottom: 20,
  },
});
