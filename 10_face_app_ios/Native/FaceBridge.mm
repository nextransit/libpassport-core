// FaceBridge.mm — Objective-C++ wrapper that exposes the C++ pipeline
// to Swift. All public methods are blocking; callers should dispatch
// to a background queue.
#import "FaceBridge.h"
#import "face_pipeline.h"

#include <memory>
#include <string>
#include <array>
#include <vector>

static std::unique_ptr<faceapp_ios::Pipeline> g_pipeline;

@implementation FaceDetection
- (instancetype)initWithX:(NSInteger)x y:(NSInteger)y w:(NSInteger)w h:(NSInteger)h score:(float)s {
    if ((self = [super init])) {
        _x = x; _y = y; _width = w; _height = h; _score = s;
    }
    return self;
}
@end

@implementation FaceFrame
- (instancetype)initWithWidth:(NSInteger)w height:(NSInteger)h faces:(NSArray<FaceDetection *> *)faces points:(NSArray<NSArray<NSNumber *> *> *)points {
    if ((self = [super init])) {
        _width = w; _height = h; _faces = [faces copy]; _points = [points copy];
    }
    return self;
}
@end

@implementation FaceBridge

+ (nullable instancetype)initWithModelDir:(NSString *)modelDir
                                          error:(NSError **)error {
    g_pipeline.reset();
    try {
        g_pipeline.reset(new faceapp_ios::Pipeline(std::string([modelDir UTF8String])));
        return (FaceBridge *)self; // succeed as a placeholder
    } catch (const std::exception& e) {
        if (error) *error = [NSError errorWithDomain:@"FaceBridge" code:1
            userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithUTF8String:e.what()]}];
        return nil;
    } catch (...) {
        if (error) *error = [NSError errorWithDomain:@"FaceBridge" code:2
            userInfo:@{NSLocalizedDescriptionKey: @"unknown error"}];
        return nil;
    }
}

+ (NSInteger)databaseSize {
    return g_pipeline ? (NSInteger)g_pipeline->DatabaseSize() : 0;
}

+ (BOOL)deleteIndex:(int64_t)index {
    return g_pipeline ? (g_pipeline->DeleteByIndex(index) == 1) : NO;
}

+ (nullable FaceFrame *)detectInNV21:(NSData *)nv21 width:(NSInteger)w height:(NSInteger)h {
    if (!g_pipeline) return nil;
    const uint8_t* bytes = (const uint8_t*)nv21.bytes;
    std::vector<faceapp_ios::DetectionOut> faces;
    std::vector<std::array<float, 10>> points;
    int n = g_pipeline->DetectAndMark(bytes, (int32_t)w, (int32_t)h, &faces, &points);
    NSMutableArray<FaceDetection *>* outFaces = [NSMutableArray arrayWithCapacity:n];
    NSMutableArray<NSArray<NSNumber *>*>* outPoints = [NSMutableArray arrayWithCapacity:n];
    for (int i = 0; i < n; ++i) {
        const auto& d = faces[i];
        FaceDetection* fd = [[FaceDetection alloc] initWithX:d.x y:d.y w:d.w h:d.h score:d.score];
        [outFaces addObject:fd];
        NSMutableArray<NSNumber *>* pa = [NSMutableArray arrayWithCapacity:10];
        for (int j = 0; j < 10; ++j) [pa addObject:@(points[i][j])];
        [outPoints addObject:pa];
    }
    return [[FaceFrame alloc] initWithWidth:w height:h faces:outFaces points:outPoints];
}

+ (int64_t)registerFromNV21:(NSData *)nv21 width:(NSInteger)w height:(NSInteger)h points:(NSArray<NSNumber *> *)points {
    if (!g_pipeline) return -1;
    if (points.count != 10) return -1;
    std::array<float, 10> p{};
    for (int i = 0; i < 10; ++i) p[i] = points[i].floatValue;
    return g_pipeline->RegisterFromNv21((const uint8_t*)nv21.bytes, (int32_t)w, (int32_t)h, p);
}

+ (int64_t)queryFromNV21:(NSData *)nv21 width:(NSInteger)w height:(NSInteger)h points:(NSArray<NSNumber *> *)points similarity:(float *)outSim {
    if (!g_pipeline) return -1;
    if (points.count != 10) return -1;
    std::array<float, 10> p{};
    for (int i = 0; i < 10; ++i) p[i] = points[i].floatValue;
    float s = 0;
    int64_t id = g_pipeline->QueryFromNv21((const uint8_t*)nv21.bytes, (int32_t)w, (int32_t)h, p, &s);
    if (outSim) *outSim = s;
    return id;
}

+ (BOOL)saveDatabaseTo:(NSString *)path {
    if (!g_pipeline) return NO;
    return g_pipeline->SaveDatabase(std::string([path UTF8String])) == 0;
}

+ (BOOL)loadDatabaseFrom:(NSString *)path {
    if (!g_pipeline) return NO;
    return g_pipeline->LoadDatabase(std::string([path UTF8String])) == 0;
}

@end
