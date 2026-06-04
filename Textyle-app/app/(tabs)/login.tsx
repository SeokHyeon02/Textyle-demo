import { FontAwesome } from '@expo/vector-icons';
import * as Google from 'expo-auth-session/providers/google';
import { router } from 'expo-router';
import * as WebBrowser from 'expo-web-browser';
import type { Session } from '@supabase/supabase-js';
import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
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
          if (error) Alert.alert('오류', error.message);
        });
    }
  }, [response]);

  const signInWithEmail = async () => {
    if (!email.trim() || !password) {
      Alert.alert('알림', '이메일과 비밀번호를 입력해주세요.');
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

  if (!session) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.container}>
          <Text style={styles.title}>Textyle 로그인</Text>
          <Text style={styles.subtitle}>검색 결과를 확인하려면 로그인해주세요.</Text>

          <TextInput
            style={styles.input}
            placeholder="이메일 주소"
            placeholderTextColor="#8E8E8E"
            value={email}
            onChangeText={setEmail}
            autoCapitalize="none"
            keyboardType="email-address"
          />
          <TextInput
            style={styles.input}
            placeholder="비밀번호"
            placeholderTextColor="#8E8E8E"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
            autoCapitalize="none"
          />

          <TouchableOpacity
            style={[styles.primaryButton, loading && styles.disabledButton]}
            onPress={signInWithEmail}
            disabled={loading}>
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.primaryButtonText}>로그인</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={() => router.push('/signup')}
            disabled={loading}>
            <Text style={styles.secondaryButtonText}>회원가입</Text>
          </TouchableOpacity>

          <View style={styles.divider}>
            <View style={styles.line} />
            <Text style={styles.orText}>또는</Text>
            <View style={styles.line} />
          </View>

          <TouchableOpacity
            style={[styles.googleButton, (!request || loading) && styles.disabledButton]}
            onPress={() => promptAsync()}
            disabled={!request || loading}>
            <FontAwesome name="google" size={18} color="#FFFFFF" style={styles.icon} />
            <Text style={styles.googleButtonText}>Google로 시작하기</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <Text style={styles.title}>내 계정</Text>
        <Text style={styles.subtitle}>검색 탭에서 이미지 검색을 시작할 수 있습니다.</Text>

        <View style={styles.profilePanel}>
          <FontAwesome name="user-circle" size={68} color="#D0D1D2" />
          <Text style={styles.email}>{session.user.email}</Text>
        </View>

        <TouchableOpacity style={styles.logoutButton} onPress={signOut}>
          <Text style={styles.logoutButtonText}>로그아웃</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  container: {
    flex: 1,
    paddingHorizontal: 20,
    justifyContent: 'center',
    backgroundColor: '#FFFFFF',
  },
  title: {
    fontSize: 28,
    fontWeight: '600',
    marginBottom: 8,
    textAlign: 'center',
    color: '#171A20',
  },
  subtitle: {
    fontSize: 15,
    lineHeight: 22,
    color: '#393C41',
    marginBottom: 28,
    textAlign: 'center',
  },
  input: {
    minHeight: 52,
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 14,
    borderRadius: 8,
    marginBottom: 12,
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
    marginTop: 8,
  },
  primaryButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  secondaryButton: {
    minHeight: 50,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#D0D1D2',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
    backgroundColor: '#FFFFFF',
  },
  secondaryButtonText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#393C41',
  },
  disabledButton: {
    opacity: 0.68,
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 22,
  },
  line: {
    flex: 1,
    height: 1,
    backgroundColor: '#EEEEEE',
  },
  orText: {
    marginHorizontal: 12,
    color: '#5C5E62',
    fontSize: 13,
  },
  googleButton: {
    minHeight: 52,
    flexDirection: 'row',
    backgroundColor: '#171A20',
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
  },
  icon: {
    marginRight: 12,
  },
  googleButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '600',
  },
  profilePanel: {
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 180,
    borderRadius: 8,
    backgroundColor: '#F4F4F4',
    marginBottom: 24,
    paddingHorizontal: 18,
  },
  email: {
    fontSize: 16,
    color: '#393C41',
    marginTop: 14,
    textAlign: 'center',
  },
  logoutButton: {
    minHeight: 50,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#F4F4F4',
  },
  logoutButtonText: {
    color: '#393C41',
    fontSize: 15,
    fontWeight: '600',
  },
});
