import { Ionicons } from '@expo/vector-icons';
import { Tabs } from 'expo-router';

// 앱 진입(로그인 후 포함) 시 기본으로 활성화되는 탭을 홈으로 지정한다.
export const unstable_settings = {
  initialRouteName: 'home',
};

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: '#3E6AE1',
        tabBarInactiveTintColor: '#5C5E62',
        headerShown: false,
        headerTitleAlign: 'center',
      }}>

      <Tabs.Screen
        name="home"
        options={{
          title: '홈',
          tabBarIcon: ({ color, focused }) =>
            <Ionicons name={focused ? "home" : "home-outline"} size={26} color={color} />,
        }}
      />

      <Tabs.Screen
        name="index"
        options={{
          title: '검색',
          tabBarIcon: ({ color, focused }) =>
            <Ionicons name={focused ? "search" : "search-outline"} size={26} color={color} />,
        }}
      />
      
      <Tabs.Screen
        name="bookmarks"
        options={{
          title: '찜',
          tabBarIcon: ({ color, focused }) => 
            <Ionicons name={focused ? "heart" : "heart-outline"} size={26} color={color} />,
        }}
      />
      
      <Tabs.Screen
        name="login"
        options={{
          title: '로그인',
          tabBarIcon: ({ color, focused }) => 
            <Ionicons name={focused ? "person" : "person-outline"} size={26} color={color} />,
        }}
      />
    </Tabs>
  );
}
