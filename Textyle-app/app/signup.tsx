import { router } from 'expo-router';
import React, { useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { supabase } from '../supabase'; // 경로 주의!

export default function SignUpScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordConfirm, setPasswordConfirm] = useState('');
  const [loading, setLoading] = useState(false);
  const [nickname, setNickname] = useState('');

  const handleSignUp = async () => {
    if (!email || !password|| !nickname) {
      Alert.alert('알림', '모든 정보를 입력해주세요.');
      return;
    }
    if (password !== passwordConfirm) {
      Alert.alert('알림', '비밀번호가 서로 일치하지 않습니다.');
      return;
    }

    setLoading(true);
    const { data, error } = await supabase.auth.signUp({
     email,
     password,
     options: {
        data: {
            nickname: nickname,
        },
     },
    });

    if (error) {
      Alert.alert('회원가입 실패', error.message);
    } else {
      Alert.alert('가입 성공!', '환영합니다. 이제 로그인해주세요.', [
        { text: '확인', onPress: () => router.replace('/login') } // 성공하면 뒤로가기(로그인 화면)
      ]);
    }
    setLoading(false);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>새 계정 만들기</Text>
      <Text style={styles.subtitle}>TexTyle에 오신 것을 환영합니다</Text>

      <TextInput
        style={styles.input}
        placeholder="이메일"
        value={email}
        onChangeText={setEmail}
        autoCapitalize="none"
        keyboardType="email-address"
      />
      <TextInput
        style={styles.input}
        placeholder="닉네임"
        value={nickname}
        onChangeText={setNickname}
      />
      <TextInput
        style={styles.input}
        placeholder="비밀번호 (6자리 이상)"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        autoCapitalize="none"
      />
      <TextInput
        style={styles.input}
        placeholder="비밀번호 확인"
        value={passwordConfirm}
        onChangeText={setPasswordConfirm}
        secureTextEntry
        autoCapitalize="none"
      />

      <TouchableOpacity style={styles.button} onPress={handleSignUp} disabled={loading}>
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>가입하기</Text>
        )}
      </TouchableOpacity>

      <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
        <Text style={styles.backButtonText}>이미 계정이 있으신가요? 로그인</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, backgroundColor: '#fff', justifyContent: 'center' },
  title: { fontSize: 28, fontWeight: 'bold', marginBottom: 10, color: '#333' },
  subtitle: { fontSize: 16, color: '#666', marginBottom: 30 },
  input: { backgroundColor: '#f5f5f5', padding: 15, borderRadius: 10, marginBottom: 15, fontSize: 16, borderWidth: 1, borderColor: '#eee' },
  button: { backgroundColor: '#8A2BE2', paddingVertical: 15, borderRadius: 10, alignItems: 'center', marginTop: 10 },
  buttonText: { fontSize: 16, fontWeight: 'bold', color: '#fff' },
  backButton: { marginTop: 20, alignItems: 'center' },
  backButtonText: { color: '#8A2BE2', fontSize: 15, fontWeight: '500' }
});