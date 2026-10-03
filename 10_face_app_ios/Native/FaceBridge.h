// FaceBridge.h
// Swift-friendly facade over the vendored SeetaFace2 C++ classes.
// All public functions are blocking and intended to be called from a
// background DispatchQueue, never on the main thread.
//
// Pixel format: every byte* argument is NV21 (Y plane + interleaved VU),
// the format AVFoundation delivers in kCVPixelFormatType_420YpCbCr8BiPlanarFullRange.
// The native side converts to RGB internally (BT.601).

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface FaceDetection : NSObject
@property (nonatomic, readonly) NSInteger x;
@property (nonatomic, readonly) NSInteger y;
@property (nonatomic, readonly) NSInteger width;
@property (nonatomic, readonly) NSInteger height;
@property (nonatomic, readonly) float score;
- (instancetype)initWithX:(NSInteger)x y:(NSInteger)y w:(NSInteger)w h:(NSInteger)h score:(float)s;
@end

@interface FaceFrame : NSObject
@property (nonatomic, readonly) NSInteger width;
@property (nonatomic, readonly) NSInteger height;
@property (nonatomic, readonly) NSArray<FaceDetection *> *faces;
@property (nonatomic, readonly) NSArray<NSArray<NSNumber *> *> *points;  // each entry: [x0,y0,x1,y1,...,x4,y4]
- (instancetype)initWithWidth:(NSInteger)w height:(NSInteger)h faces:(NSArray<FaceDetection *> *)faces points:(NSArray<NSArray<NSNumber *> *> *)points;
@end

@interface FaceBridge : NSObject

// NS_SWIFT_NAME(init(modelDir:)) lets Swift call this as
// `try FaceBridge(modelDir: ...)` and propagate errors via throws.
+ (nullable instancetype)initWithModelDir:(NSString *)modelDir
                                   error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(init(modelDir:));
+ (NSInteger)databaseSize;
+ (BOOL)deleteIndex:(int64_t)index;

+ (nullable FaceFrame *)detectInNV21:(NSData *)nv21 width:(NSInteger)w height:(NSInteger)h;
+ (int64_t)registerFromNV21:(NSData *)nv21 width:(NSInteger)w height:(NSInteger)h points:(NSArray<NSNumber *> *)points;
+ (int64_t)queryFromNV21:(NSData *)nv21 width:(NSInteger)w height:(NSInteger)h points:(NSArray<NSNumber *> *)points similarity:(float * _Nullable)outSim;

+ (BOOL)saveDatabaseTo:(NSString *)path;
+ (BOOL)loadDatabaseFrom:(NSString *)path;

@end

NS_ASSUME_NONNULL_END
