# comfyui-mesh-bridge

ComfyUI custom nodes used by the One The Lab workflow **"그림 한 장으로 게임 몬스터 완성"**
(Hunyuan3D 2.1 → low-poly → Paint 2.1 texture → normal bake → auto rig → Unity FBX).

- Workflow page: https://onethelab.com/01_public/03_ai/10_comfy/01_workflow/05_03_Hunyuan3D_game_model
- YouTube: One The Lab (@onethelab)

## 설치 (Install)

**ComfyUI Manager** → `Install via Git URL` → 아래 주소 입력 → ComfyUI 재시작

```
https://github.com/gunug/comfyui-mesh-bridge
```

또는 직접:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gunug/comfyui-mesh-bridge
```

필요한 패키지는 `numpy`, `trimesh`, `shapely` 입니다(ComfyUI-Hunyuan3d-2-1 을 설치했다면 numpy·trimesh 는 이미 있습니다).
SVG 돋을새김 노드는 `manifold3d` 가 있으면 조각을 하나로 합쳐 watertight STL 을 냅니다(없으면 단순 합치기).
없으면 ComfyUI 의 파이썬으로 `pip install -r requirements.txt`.

## 노드 (Nodes)

| 노드 | 하는 일 |
| --- | --- |
| MESH → TRIMESH (bridge) | ComfyUI 기본 MESH 를 trimesh(TRIMESH) 로. Hunyuan3D 2.1 Paint 노드에 넣을 때 |
| TRIMESH → MESH (bridge) | 반대 방향. UV 와 텍스처를 같이 넘긴다 |
| MESH → TRIMESH (PBR 텍스처 유지) | 텍스처 입힌 MESH 를 UV·PBR 맵째로 TRIMESH 로 (SaveGLB 와 같은 직렬화) |
| 노멀맵 G 반전 (OpenGL↔DirectX) | 노멀맵 G 채널 반전. Unity 는 켜고 Unreal 은 끈다 |
| 노멀맵 강도 (굴곡 깊이) | 노멀맵 XY 를 배율로 키우고 Z 를 다시 계산. 1 = 원본 |
| 유니티 마스크맵 패킹 | R=메탈릭 G=AO B=디테일 A=스무스니스 |
| 유니티 에셋 이름 (다음 버전 번호) | `output/<prefix>_v001`, `_v002` ... 다음 빈 이름 |
| 유니티 리깅 FBX 최종 출력 | 리깅된 FBX 를 `output/<prefix>_vNNN.fbx` 로 옮기고 부산물(.fbm, 미리보기 glb) 정리 |
| TRIMESH / MESH → 이미지 렌더 | OpenGL 없이 numpy 로 턴테이블 이미지 렌더 |
| SVG Relief to Mesh (extrude + base) | SVG(vtracer 채움 패스)의 어두운 부분을 돋을새김 + 받침판 MESH 로. mm 단위, 좌우 반전(도장), 빼기 기울기(draft) |
| Save Mesh as STL | MESH 를 `output/<prefix>_NNNNN_.stl` 로 저장 (3D 프린트용) |

## 텍스처 이름 (선택)

ComfyUI-UniRig 원본으로 리깅하면 FBX 안의 텍스처 이름이 `Image_0`, `Image_1` 로 나옵니다.
**리깅과 Unity 적용에는 문제가 없습니다.** 다만 몬스터 여러 마리를 Unity 의 같은 폴더에
Extract Textures 하면 이름이 겹칩니다.

`<에셋>_albedo` / `<에셋>_normal` 로 나오게 하려면 `patches/unirig-texture-names.patch` 를
ComfyUI-UniRig 에 적용합니다(기준 커밋은 `patches/UNIRIG_BASE_COMMIT`).

```bash
cd ComfyUI/custom_nodes/ComfyUI-UniRig
git apply ../comfyui-mesh-bridge/patches/unirig-texture-names.patch
```

UniRig 를 업데이트하면 패치가 지워지므로 다시 적용합니다.

## License

MIT. Hunyuan3D 2.1 등 이 노드와 함께 쓰는 모델은 각자의 라이선스를 따릅니다
(Tencent Hunyuan 3D 2.1 Community License 는 한국 · EU · 영국을 적용 지역에서 제외합니다).
