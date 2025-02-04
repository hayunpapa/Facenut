<%@ page import="java.io.FileOutputStream, java.io.IOException" %>
<%@ page import="java.util.Base64" %>
<%@ page contentType="text/html; charset=UTF-8" %>
<%
    // 1) 요청 파라미터에서 Base64 이미지 문자열 받기
    String base64Image = request.getParameter("imageData");
    if (base64Image == null || base64Image.isEmpty()) {
        out.println("이미지 데이터가 전송되지 않았습니다.");
        return;
    }

    // 2) data:image/png;base64, ... 형태인지 확인
    //    일반적으로 "data:image/png;base64," 부분을 제거하고 실제 Base64 데이터만 추출
    String base64Prefix = "data:image/png;base64,";  // png 기준 예시
    // 만약 JPEG, GIF 등 여러 타입일 경우 아래와 같이 확장:
    // String base64PrefixJpg = "data:image/jpeg;base64,";
    // ...
    
    if (!base64Image.startsWith(base64Prefix)) {
        out.println("잘못된 형식의 이미지 데이터입니다.");
        return;
    }

    // 3) Base64 데이터만 추출하기
    String pureBase64 = base64Image.substring(base64Prefix.length());

    // 4) 디코딩 (byte[] 형태로 변환)
    byte[] imageBytes = null;
    try {
        imageBytes = Base64.getDecoder().decode(pureBase64);
    } catch (IllegalArgumentException e) {
        out.println("Base64 디코딩 중 오류가 발생했습니다: " + e.getMessage());
        return;
    }

    // 5) 서버에 파일로 저장
    //    실제 운영에선 다른 식별자(로그인한 사용자, 날짜/시간 등)로 파일명을 정해줄 수 있음
    String fileName = "upload_" + System.currentTimeMillis() + ".png";
    String savePath = "d:/uploads/";   // 저장 경로 (운영 환경에 맞춰 변경)
    
    // 저장 폴더가 없으면 생성
    java.io.File folder = new java.io.File(savePath);
    if (!folder.exists()) {
        folder.mkdirs();
    }

    // 파일 출력 스트림을 이용해 저장
    try (FileOutputStream fos = new FileOutputStream(savePath + fileName)) {
        fos.write(imageBytes);
        fos.flush();
    } catch (IOException e) {
        out.println("이미지 파일 저장 중 오류가 발생했습니다: " + e.getMessage());
        return;
    }

    // 6) 처리 결과
    out.println("<h3>이미지 업로드 성공</h3>");
    out.println("저장된 파일명: " + fileName + "<br/>");
    out.println("저장된 경로: " + savePath + "<br/>");
%>
