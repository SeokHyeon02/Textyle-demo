import { FontAwesome, Ionicons } from '@expo/vector-icons';
import type { Session } from '@supabase/supabase-js';
import * as Google from 'expo-auth-session/providers/google';
import { router } from 'expo-router';
import * as WebBrowser from 'expo-web-browser';
import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { supabase } from '../../supabase';

WebBrowser.maybeCompleteAuthSession();

export default function LoginScreen() {
  const [session, setSession] = useState<Session | null>(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [showEmailForm, setShowEmailForm] = useState(false);
  const [showAuthOptions, setShowAuthOptions] = useState(false);

  const [request, response, promptAsync] = Google.useAuthRequest({
    webClientId: '509294193303-6fc0fgvftk04hb7l0frqta6lmmejdoop.apps.googleusercontent.com',
    iosClientId: '509294193303-km6ho5gcvu02cfhiqlurc2dbppindte3.apps.googleusercontent.com',
  });

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setSession(session));
    const { data } = supabase.auth.onAuthStateChange((_event, session) => setSession(session));

    return () => data.subscription.unsubscribe();
  }, []);

  useEffect(() => {
    if (response?.type === 'success') {
      const { id_token } = response.params;
      supabase.auth.signInWithIdToken({ provider: 'google', token: id_token })
        .then(({ error }) => {
          if (error) Alert.alert('로그인 실패', error.message);
        });
    }
  }, [response]);

  const signInWithEmail = async () => {
    if (!email.trim() || !password) {
      Alert.alert('입력 필요', '이메일과 비밀번호를 입력해주세요.');
      return;
    }

    setLoading(true);
    const { error } = await supabase.auth.signInWithPassword({
      email: email.trim(),
      password,
    });

    if (error) {
      Alert.alert('로그인 실패', error.message);
    }
    setLoading(false);
  };

  const signOut = async () => {
    await supabase.auth.signOut();
  };

  const showDeleteAccountNotice = () => {
    Alert.alert('회원탈퇴', '회원탈퇴 기능은 계정 삭제 정책 확인 후 연결할 예정입니다.');
  };

  if (!session) {
    if (showEmailForm) {
      return (
        <SafeAreaView style={styles.safeArea}>
          <KeyboardAvoidingView
            style={styles.keyboardContainer}
            behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
            <ScrollView
              contentContainerStyle={styles.emailScreenContent}
              keyboardShouldPersistTaps="handled"
              showsVerticalScrollIndicator={false}>
              <TouchableOpacity
                style={styles.backIconButton}
                onPress={() => setShowEmailForm(false)}
                activeOpacity={0.72}>
                <Ionicons name="chevron-back" size={28} color="#171A20" />
              </TouchableOpacity>

              <View style={styles.emailScreenHeader}>
                <Text style={styles.emailScreenTitle}>이메일로 로그인</Text>
                <Text style={styles.emailScreenSubtitle}>가입한 이메일과 비밀번호를 입력해주세요.</Text>
              </View>

              <View style={styles.emailForm}>
                <TextInput
                  style={styles.input}
                  placeholder="이메일 주소"
                  placeholderTextColor="#8E8E8E"
                  value={email}
                  onChangeText={setEmail}
                  autoCapitalize="none"
                  keyboardType="email-address"
                  textContentType="emailAddress"
                />
                <TextInput
                  style={styles.input}
                  placeholder="비밀번호"
                  placeholderTextColor="#8E8E8E"
                  value={password}
                  onChangeText={setPassword}
                  secureTextEntry
                  autoCapitalize="none"
                  textContentType="password"
                />
                <TouchableOpacity
                  style={styles.signUpInlineButton}
                  onPress={() => router.push('/signup')}
                  disabled={loading}
                  activeOpacity={0.78}>
                  <Text style={styles.signUpInlineText}>계정이 없으신가요? 회원가입</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.primaryButton, loading && styles.disabledButton]}
                  onPress={signInWithEmail}
                  disabled={loading}
                  activeOpacity={0.84}>
                  {loading ? (
                    <ActivityIndicator color="#FFFFFF" />
                  ) : (
                    <Text style={styles.primaryButtonText}>로그인</Text>
                  )}
                </TouchableOpacity>
              </View>
            </ScrollView>
          </KeyboardAvoidingView>
        </SafeAreaView>
      );
    }

    return (
      <SafeAreaView style={styles.safeArea}>
        <KeyboardAvoidingView
          style={styles.keyboardContainer}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
          <ScrollView
            contentContainerStyle={styles.authContent}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}>
            <View style={styles.guestHero}>
              <Text style={styles.guestTitle}>
                이미지 검색의 시작{'\n'}
                Textyle과 함께하세요!
              </Text>
              <TouchableOpacity
                style={styles.guestLoginButton}
                onPress={() => setShowAuthOptions(prev => !prev)}
                activeOpacity={0.84}>
                <Text style={styles.guestLoginButtonText}>로그인 / 회원가입</Text>
              </TouchableOpacity>
            </View>

            {showAuthOptions && (
              <View style={styles.actionBlock}>
                <TouchableOpacity
                  style={[styles.googleButton, (!request || loading) && styles.disabledButton]}
                  onPress={() => promptAsync()}
                  disabled={!request || loading}
                  activeOpacity={0.84}>
                  <FontAwesome name="google" size={18} color="#FFFFFF" style={styles.buttonIcon} />
                  <Text style={styles.googleButtonText}>Google로 계속하기</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.emailEntryButton}
                  onPress={() => setShowEmailForm(true)}
                  disabled={loading}
                  activeOpacity={0.84}>
                  <Ionicons name="mail-outline" size={19} color="#171A20" style={styles.buttonIcon} />
                  <Text style={styles.emailEntryButtonText}>이메일로 로그인</Text>
                  <Ionicons name="chevron-forward" size={18} color="#5C5E62" style={styles.chevronIcon} />
                </TouchableOpacity>
              </View>
            )}

            <View style={styles.guestMenu}>
              <TouchableOpacity style={styles.guestMenuRow} activeOpacity={0.72}>
                <Text style={styles.guestMenuText}>공지사항</Text>
                <Ionicons name="chevron-forward" size={18} color="#D0D1D2" />
              </TouchableOpacity>
              <TouchableOpacity style={styles.guestMenuRow} activeOpacity={0.72}>
                <Text style={styles.guestMenuText}>고객센터</Text>
                <Ionicons name="chevron-forward" size={18} color="#D0D1D2" />
              </TouchableOpacity>
              <TouchableOpacity style={styles.guestMenuRow} activeOpacity={0.72}>
                <Text style={styles.guestMenuText}>앱 설정</Text>
                <Ionicons name="chevron-forward" size={18} color="#D0D1D2" />
              </TouchableOpacity>
            </View>
          </ScrollView>
        </KeyboardAvoidingView>
      </SafeAreaView>
    );
  }

  const displayName =
    session.user.user_metadata?.nickname ||
    session.user.user_metadata?.name ||
    session.user.email?.split('@')[0] ||
    'Textyle';

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.profileContent} showsVerticalScrollIndicator={false}>
        <View style={styles.memberHero}>
          <Text style={styles.memberName}>{displayName} 님</Text>
          <View style={styles.memberSummaryCard}>
            <View style={styles.memberAvatarBox}>
              <FontAwesome name="user-circle" size={58} color="#8E8E8E" />
              <Text style={styles.memberAvatarLabel}>회원정보</Text>
            </View>
            <View style={styles.memberInfoBox}>
              <View style={styles.memberInfoRow}>
                <Text style={styles.memberInfoLabel}>이메일</Text>
                <Text style={styles.memberInfoValue} numberOfLines={1}>
                  {session.user.email || '이메일 정보 없음'}
                </Text>
              </View>
              <View style={styles.memberDivider} />
              <View style={styles.memberInfoRow}>
                <Text style={styles.memberInfoLabel}>가입 방식</Text>
                <Text style={styles.memberInfoValue}>
                  {session.user.app_metadata?.provider === 'google' ? 'Google' : '이메일'}
                </Text>
              </View>
            </View>
          </View>
        </View>

        <View style={styles.memberMenu}>
          <TouchableOpacity style={styles.memberMenuRow} activeOpacity={0.72}>
            <Text style={styles.memberMenuText}>회원정보</Text>
            <Ionicons name="chevron-forward" size={18} color="#D0D1D2" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.memberMenuRow} activeOpacity={0.72}>
            <Text style={styles.memberMenuText}>북마크</Text>
            <Ionicons name="chevron-forward" size={18} color="#D0D1D2" />
          </TouchableOpacity>
        </View>

        <View style={styles.memberMenu}>
          <TouchableOpacity style={styles.memberMenuRow} activeOpacity={0.72}>
            <Text style={styles.memberMenuText}>친구초대</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.memberMenuRow} activeOpacity={0.72}>
            <Text style={styles.memberMenuText}>공지사항</Text>
            <Ionicons name="chevron-forward" size={18} color="#D0D1D2" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.memberMenuRow} activeOpacity={0.72}>
            <Text style={styles.memberMenuText}>고객센터</Text>
            <Ionicons name="chevron-forward" size={18} color="#D0D1D2" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.memberMenuRow} activeOpacity={0.72}>
            <Text style={styles.memberMenuText}>앱 설정</Text>
            <Ionicons name="chevron-forward" size={18} color="#D0D1D2" />
          </TouchableOpacity>
        </View>

        <View style={styles.memberMenu}>
          <TouchableOpacity style={styles.memberMenuRow} onPress={signOut} activeOpacity={0.72}>
            <Text style={styles.memberMenuText}>로그아웃</Text>
            <Ionicons name="chevron-forward" size={18} color="#D0D1D2" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.memberMenuRow} onPress={showDeleteAccountNotice} activeOpacity={0.72}>
            <Text style={styles.deleteMenuText}>회원탈퇴</Text>
            <Ionicons name="chevron-forward" size={18} color="#D0D1D2" />
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  keyboardContainer: {
    flex: 1,
  },
  authContent: {
    flexGrow: 1,
    paddingTop: 82,
    paddingBottom: 36,
  },
  guestHero: {
    paddingHorizontal: 28,
    paddingBottom: 38,
    borderBottomWidth: 8,
    borderBottomColor: '#F4F4F4',
  },
  guestTitle: {
    color: '#171A20',
    fontSize: 26,
    lineHeight: 36,
    fontWeight: '600',
    marginBottom: 28,
  },
  guestLoginButton: {
    minHeight: 62,
    borderRadius: 8,
    backgroundColor: '#3E6AE1',
    alignItems: 'center',
    justifyContent: 'center',
  },
  guestLoginButtonText: {
    color: '#FFFFFF',
    fontSize: 17,
    lineHeight: 23,
    fontWeight: '600',
  },
  actionBlock: {
    width: '100%',
    maxWidth: 430,
    alignSelf: 'center',
    gap: 12,
    paddingHorizontal: 28,
    paddingTop: 22,
  },
  googleButton: {
    minHeight: 54,
    borderRadius: 6,
    backgroundColor: '#171A20',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
  },
  googleButtonText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '600',
  },
  emailEntryButton: {
    minHeight: 54,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#D0D1D2',
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
  },
  emailEntryButtonText: {
    color: '#171A20',
    fontSize: 13,
    fontWeight: '600',
  },
  buttonIcon: {
    marginRight: 12,
  },
  chevronIcon: {
    position: 'absolute',
    right: 16,
  },
  emailForm: {
    gap: 10,
    marginTop: 2,
  },
  emailScreenContent: {
    flexGrow: 1,
    paddingHorizontal: 28,
    paddingTop: 18,
    paddingBottom: 36,
  },
  backIconButton: {
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: -12,
  },
  emailScreenHeader: {
    marginTop: 52,
    marginBottom: 28,
  },
  emailScreenTitle: {
    color: '#171A20',
    fontSize: 30,
    lineHeight: 38,
    fontWeight: '600',
  },
  emailScreenSubtitle: {
    marginTop: 10,
    color: '#5C5E62',
    fontSize: 14,
    lineHeight: 20,
  },
  signUpInlineButton: {
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
    backgroundColor: '#F4F4F4',
  },
  signUpInlineText: {
    color: '#393C41',
    fontSize: 14,
    lineHeight: 19,
    fontWeight: '600',
  },
  guestMenu: {
    paddingHorizontal: 28,
    paddingTop: 28,
  },
  guestMenuRow: {
    minHeight: 64,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  guestMenuText: {
    color: '#171A20',
    fontSize: 19,
    lineHeight: 25,
    fontWeight: '600',
  },
  kicker: {
    color: '#3E6AE1',
    fontSize: 22,
    lineHeight: 28,
    fontWeight: '600',
    marginBottom: 18,
  },
  input: {
    minHeight: 52,
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 14,
    borderRadius: 6,
    fontSize: 15,
    borderWidth: 1,
    borderColor: '#EEEEEE',
    color: '#171A20',
  },
  primaryButton: {
    minHeight: 52,
    borderRadius: 6,
    backgroundColor: '#3E6AE1',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 2,
  },
  primaryButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  disabledButton: {
    opacity: 0.68,
  },
  profileContent: {
    flexGrow: 1,
    backgroundColor: '#FFFFFF',
    paddingBottom: 42,
  },
  memberHero: {
    paddingHorizontal: 24,
    paddingTop: 86,
    paddingBottom: 24,
    backgroundColor: '#F4F4F4',
  },
  memberName: {
    color: '#171A20',
    fontSize: 31,
    lineHeight: 39,
    fontWeight: '600',
    marginBottom: 20,
  },
  memberSummaryCard: {
    flexDirection: 'row',
    gap: 12,
  },
  memberAvatarBox: {
    width: 122,
    minHeight: 128,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  memberAvatarLabel: {
    color: '#8E8E8E',
    fontSize: 13,
    lineHeight: 19,
    fontWeight: '600',
    marginTop: 10,
  },
  memberInfoBox: {
    flex: 1,
    minHeight: 128,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 16,
    justifyContent: 'center',
  },
  memberInfoRow: {
    minHeight: 48,
    justifyContent: 'center',
  },
  memberInfoLabel: {
    color: '#8E8E8E',
    fontSize: 10,
    lineHeight: 15,
    marginBottom: 3,
  },
  memberInfoValue: {
    color: '#171A20',
    fontSize: 13,
    lineHeight: 19,
    fontWeight: '600',
  },
  memberDivider: {
    height: 1,
    backgroundColor: '#EEEEEE',
  },
  memberMenu: {
    paddingHorizontal: 24,
    paddingVertical: 18,
    borderBottomWidth: 8,
    borderBottomColor: '#F4F4F4',
  },
  memberMenuRow: {
    minHeight: 58,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  memberMenuText: {
    color: '#171A20',
    fontSize: 18,
    lineHeight: 25,
    fontWeight: '600',
  },
  deleteMenuText: {
    color: '#B42318',
    fontSize: 18,
    lineHeight: 25,
    fontWeight: '600',
  },
});
